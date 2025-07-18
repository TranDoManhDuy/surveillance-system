import torch
import cv2
import time
import numpy as np
import argparse
import os
import math
import threading
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
from queue import Queue, Empty
import logging
from scipy.spatial import KDTree
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreadedPersonTracker:
    def __init__(self, model_name='yolov5m', confidence_threshold=0.5, device=None, 
                 batch_size=4, fps=30):
        """
        Initialize the threaded person tracker with distance monitoring
        
        Args:
            model_name (str): YOLOv5 model variant
            confidence_threshold (float): Minimum confidence for detection
            device (str): Device to run on
            batch_size (int): Batch size for processing
            fps (int): Target FPS for processing
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.fps = fps
        
        # Load YOLOv5 model
        print(f"Loading {model_name} model on {self.device}...")
        self.model = torch.hub.load('ultralytics/yolov5', model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Tracking variables
        self.tracks = {}  # {track_id: Track object}
        self.next_id = 1
        self.max_disappeared = 30  # Max frames before removing track
        self.max_distance = 100   # Max distance for matching detections
        
        # Distance monitoring parameters
        self.PERSON_HEIGHT_REAL = 1.7  # meters
        self.SOCIAL_DISTANCE_THRESHOLD = 2.0  # meters
        self.WARNING_DURATION = 1  # seconds
        
        # Distance tracking
        self.distance_history = defaultdict(lambda: deque(maxlen=3000))
        self.warned_pairs = set()
        
        # Threading components
        self.frame_queue = Queue(maxsize=self.fps * 0.5)  # Buffer for 0.5 seconds
        self.display_queue = Queue(maxsize=self.fps * 0.5)  # Buffer for display
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0
        self.current_fps = fps
        
        # Colors for different IDs
        self.colors = self._generate_colors(50)
        self.available_ids = deque()
        
        # Thread locks
        self.tracks_lock = threading.Lock()
        self.stats_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'detection_time': 0,
            'tracking_time': 0,
            'active_tracks': 0,
            'close_pairs': 0,
            'warnings_issued': 0
        }
    def _generate_colors(self, n):
        """Generate n distinct colors for tracking visualization"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = tuple(int(c * 255) for c in self._hsv_to_rgb(hue, 0.8, 0.9))
            colors.append(rgb)
        return colors
    
    def _hsv_to_rgb(self, h, s, v):
        """Convert HSV to RGB"""
        import colorsys
        return colorsys.hsv_to_rgb(h, s, v)
    
    def set_fps(self, fps):
        """Set the video FPS for accurate timing"""
        self.current_fps = fps
        max_frames = int(fps * self.WARNING_DURATION)
        self.distance_history = defaultdict(lambda: deque(maxlen=max_frames))

    def capture_loop(self, video_path):
        """
        Capture frames from video and add to queue
        Runs in separate thread
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video file {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.set_fps(fps)
        
        frame_num = 0
        
        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                
                # Flip frame if needed
                frame = cv2.flip(frame, 1)
                
                # Add frame to queue (non-blocking)
                if not self.frame_queue.full():
                    self.frame_queue.put((frame_num, frame))
                
                # Control frame rate
                time.sleep(1/self.fps)
                
        except Exception as e:
            logger.error(f"Error in capture loop: {e}")
        finally:
            cap.release()
            logger.info("Capture loop ended")
            
            self.frame_queue.put(None)

    def process_loop(self):
        """
        Process frames from queue and run detection/tracking
        Runs in separate thread
        """
        batch = []
        
        while not self.stop_event.is_set():
            try:
                # Collect frames for batch processing
                while not self.frame_queue.empty() and len(batch) < self.batch_size:
                    frame_data = self.frame_queue.get(timeout=0.1)
                    if frame_data is None:
                        self.stop_event.set()
                        break
                    batch.append(frame_data)
                
                if batch:
                    # Process batch
                    results = self._process_batch(batch)
                    
                    # Add results to display queue
                    for result in results:
                        if not self.display_queue.full():
                            self.display_queue.put(result)
                    
                    batch = []
                
                # Small delay to prevent CPU overload
                time.sleep(1/self.fps)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                
        logger.info("Process loop ended")

    def _process_batch(self, batch):
        """Process a batch of frames"""
        results = []
        
        for frame_num, frame in batch:
            start_time = time.time()
            
            # Detect persons
            detections, detection_time = self.detect_persons(frame)
            
            # Update tracks
            tracking_start = time.time()
            with self.tracks_lock:
                self.update_tracks(detections)
                close_pairs = self.monitor_distances()
            tracking_time = time.time() - tracking_start
            
            # Draw results
            annotated_frame = self.draw_tracks(frame.copy(), close_pairs)
            
            # Update statistics
            with self.stats_lock:
                self.stats['frames_processed'] += 1
                self.stats['detection_time'] += detection_time
                self.stats['tracking_time'] += tracking_time
                self.stats['active_tracks'] = sum(1 for track in self.tracks.values() if track.disappeared == 0)
                self.stats['close_pairs'] = len(close_pairs)
                self.stats['warnings_issued'] = len(self.warned_pairs)
            
            total_time = time.time() - start_time
            
            results.append({
                'frame_num': frame_num,
                'frame': annotated_frame,
                'detections': len(detections),
                'close_pairs': len(close_pairs),
                'processing_time': total_time
            })
        
        return results

    def display_loop(self):
        """
        Display processed frames
        Runs in main thread
        """
        while not self.stop_event.is_set():
            try:
                if not self.display_queue.empty():
                    result = self.display_queue.get(timeout=0.1)
                    
                    frame = result['frame']
                    frame_num = result['frame_num']
                    
                    # Add performance info
                    self._add_performance_overlay(frame, result)
                    
                    # Display frame
                    cv2.imshow("Multi-threaded Person Tracking", frame)
                    
                    # Handle key events
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.stop_event.set()
                        break
                    elif key == ord('s'):
                        cv2.imwrite(f"tracking_frame_{frame_num}.jpg", frame)
                        print(f"Saved frame {frame_num}")
                    elif key == ord('r'):  # Reset tracking
                        with self.tracks_lock:
                            self.tracks = {}
                            self.next_id = 1
                            self.distance_history.clear()
                            self.warned_pairs.clear()
                        print("Tracking reset!")
                    elif key == ord('i'):  # Show info
                        self._print_statistics()
                
                # Control display rate
                time.sleep(1/self.fps)
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in display loop: {e}")
            if self.stop_event.is_set() and self.display_queue.empty():
                break
        
        cv2.destroyAllWindows()
        logger.info("Display loop ended")

    def _add_performance_overlay(self, frame, result):
        """Add performance information overlay to frame"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Calculate FPS
        current_fps = 1.0 / result['processing_time'] if result['processing_time'] > 0 else 0
        
        info_text = [
            f"Frame: {result['frame_num']}",
            f"Processing FPS: {current_fps:.1f}",
            f"Active IDs: {stats['active_tracks']}",
            f"Detections: {result['detections']}",
            f"Close pairs: {result['close_pairs']}",
            f"Warnings: {stats['warnings_issued']}",
            f"Queue sizes: F:{self.frame_queue.qsize()}, D:{self.display_queue.qsize()}"
        ]
        
        # Add text with background
        for i, text in enumerate(info_text):
            y_pos = 30 + i * 25
            
            # Background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (5, y_pos - 20), (text_size[0] + 15, y_pos + 5), (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _print_statistics(self):
        """Print detailed statistics"""
        with self.stats_lock:
            stats = self.stats.copy()
        
        if stats['frames_processed'] > 0:
            avg_detection_time = stats['detection_time'] / stats['frames_processed']
            avg_tracking_time = stats['tracking_time'] / stats['frames_processed']
            
            print(f"\n=== TRACKING STATISTICS ===")
            print(f"Frames processed: {stats['frames_processed']}")
            print(f"Average detection time: {avg_detection_time:.3f}s")
            print(f"Average tracking time: {avg_tracking_time:.3f}s")
            print(f"Active tracks: {stats['active_tracks']}")
            print(f"Close pairs: {stats['close_pairs']}")
            print(f"Warnings issued: {stats['warnings_issued']}")
            print(f"Queue sizes - Frame: {self.frame_queue.qsize()}, Display: {self.display_queue.qsize()}")
            
            if self.warned_pairs:
                print("Warning pairs:")
                for pair in self.warned_pairs:
                    print(f"  - ID {pair[0]} and ID {pair[1]}")

    def detect_persons(self, frame):
        """Detect persons in frame"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        results = self.model(img, size=640)
        inference_time = time.time() - start_time
        
        # Extract person detections
        detections = []
        preds = results.pred[0]
        
        for *xyxy, conf, cls in preds:
            if int(cls) == 0 and conf > self.confidence_threshold:  # Person class
                x1, y1, x2, y2 = map(int, xyxy)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': (center_x, center_y),
                    'confidence': float(conf),
                    'area': width * height,
                    'height_pixels': height
                })
        
        return detections, inference_time

    def calculate_pixel_distance(self, center1, center2):
        """Calculate Euclidean distance between two centers in pixels"""
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_real_distance(self, center1, center2, height1_pixels, height2_pixels):
        """Calculate real-world distance between two persons"""
        avg_height_pixels = (height1_pixels + height2_pixels) / 2
        pixels_per_meter = avg_height_pixels / self.PERSON_HEIGHT_REAL
        pixel_distance = self.calculate_pixel_distance(center1, center2)
        real_distance = pixel_distance / pixels_per_meter
        return real_distance
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def update_tracks(self, detections):
        """Update tracks using Hungarian algorithm"""
        track_ids = list(self.tracks.keys())
        num_tracks = len(track_ids)
        num_detections = len(detections)

        if num_tracks == 0:
            for detection in detections:
                self.tracks[self.next_id] = Track(self.next_id, detection)
                self.next_id += 1
            return

        cost_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, detection in enumerate(detections):
                distance = self.calculate_pixel_distance(track.center, detection['center'])
                iou = self.calculate_iou(track.bbox, detection['bbox'])
                cost = distance - (iou * 50)
                cost_matrix[i, j] = cost

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_tracks = set()
        assigned_detections = set()

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.max_distance:
                track_id = track_ids[i]
                self.tracks[track_id].update(detection=detections[j])
                assigned_tracks.add(track_id)
                assigned_detections.add(j)

        # Update unmatched tracks
        for i, track_id in enumerate(track_ids):
            if track_id not in assigned_tracks:
                self.tracks[track_id].disappeared += 1

        # Add new tracks
        for j, detection in enumerate(detections):
            if j not in assigned_detections:
                if self.available_ids:
                    reuse_id = self.available_ids.popleft()
                else:
                    reuse_id = self.next_id
                    self.next_id += 1
                self.tracks[reuse_id] = Track(reuse_id, detection)

        # Remove disappeared tracks
        to_remove = [track_id for track_id, t in self.tracks.items() if t.disappeared > self.max_disappeared]
        for track_id in to_remove:
            del self.tracks[track_id]
            self.available_ids.append(track_id)

    def monitor_distances(self):
        active_tracks = [(tid, track) for tid, track in self.tracks.items() if track.disappeared == 0]
        if len(active_tracks) < 2:
            return []

        centers = [track.center for _, track in active_tracks]
        tree = KDTree(centers)
        pairs = tree.query_pairs(r=self.SOCIAL_DISTANCE_THRESHOLD * 100)  # Ước lượng pixel
        close_pairs = []

        for i, j in pairs:
            id1, track1 = active_tracks[i]
            id2, track2 = active_tracks[j]
            real_distance = self.calculate_real_distance(track1.center, track2.center, 
                                                        track1.height_pixels, track2.height_pixels)
            if real_distance < self.SOCIAL_DISTANCE_THRESHOLD:
                close_pairs.append((id1, id2, real_distance))
                # Xử lý lịch sử và cảnh báo như hiện tại
        return close_pairs

    def draw_tracks(self, frame, close_pairs):
        """Draw tracking results on frame"""
        for track_id, track in self.tracks.items():
            if track.disappeared > 0:
                continue
            
            x1, y1, x2, y2 = track.bbox
            color = self.colors[track_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track trail
            if len(track.trail) > 1:
                for i in range(1, len(track.trail)):
                    cv2.line(frame, track.trail[i-1], track.trail[i], color, 2)
            
            # Draw label
            label = f'ID: {track_id} ({track.confidence:.2f})'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.circle(frame, track.center, 3, color, -1)
        
        # Draw distance lines
        for id1, id2, distance in close_pairs:
            if id1 in self.tracks and id2 in self.tracks:
                track1 = self.tracks[id1]
                track2 = self.tracks[id2]
                
                cv2.line(frame, track1.center, track2.center, (0, 0, 255), 2)
                
                mid_x = (track1.center[0] + track2.center[0]) // 2
                mid_y = (track1.center[1] + track2.center[1]) // 2
                
                distance_text = f'{distance:.1f}m'
                cv2.putText(frame, distance_text, (mid_x, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return frame

    def run(self, video_path):
        """
        Run the multi-threaded tracking system
        """
        print(f"Starting multi-threaded person tracking on {video_path}")
        print(f"Using device: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print(f"Target FPS: {self.fps}")
        
        # Start threads
        capture_thread = threading.Thread(target=self.capture_loop, args=(video_path,))
        process_thread = threading.Thread(target=self.process_loop)
        
        capture_thread.start()
        process_thread.start()
        
        # Run display loop in main thread
        try:
            self.display_loop()
        except KeyboardInterrupt:
            print("\nStopping...")
        
        # Stop all threads
        self.stop_event.set()
        
        # Wait for threads to finish
        capture_thread.join()
        process_thread.join()
        
        # Final statistics
        self._print_statistics()
        print("Multi-threaded tracking completed!")


class Track:
    def __init__(self, track_id, detection):
        self.id = track_id
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = detection['confidence']
        self.height_pixels = detection['height_pixels']
        self.smoothed_height = detection['height_pixels']
        self.alpha = 0.6
        self.disappeared = 0
        self.trail = [detection['center']]
        self.max_trail_length = 20
        self.age = 0
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0
    
    def update(self, detection):
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = detection['confidence']
        self.smoothed_height = self.alpha * detection['height_pixels'] + (1 - self.alpha) * self.smoothed_height
        self.height_pixels = self.smoothed_height
        self.disappeared = 0
        self.age += 1
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0
        
        self.trail.append(detection['center'])
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)


def main():
    """Main function to run the threaded tracker"""
    parser = argparse.ArgumentParser(description='Multi-threaded Person Tracking with Distance Monitoring')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--model', type=str, default='yolov5s', help='YOLOv5 model variant')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--device', type=str, default=None, help='Device to run on')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for processing')
    parser.add_argument('--fps', type=int, default=30, help='Target FPS')
    parser.add_argument('--social-distance', type=float, default=2.0, help='Social distance threshold in meters')
    parser.add_argument('--warning-time', type=float, default=1.0, help='Warning time threshold in seconds')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        return
    
    # Create tracker
    tracker = ThreadedPersonTracker(
        model_name=args.model,
        confidence_threshold=args.conf,
        device=args.device,
        batch_size=args.batch_size,
        fps=args.fps
    )
    
    # Update thresholds
    tracker.SOCIAL_DISTANCE_THRESHOLD = args.social_distance
    tracker.WARNING_DURATION = args.warning_time
    
    # Run tracking
    tracker.run(args.video)


if __name__ == "__main__":
    # Example usage
    video_path = 0
    
    if os.path.exists(video_path):
        tracker = ThreadedPersonTracker(
            confidence_threshold=0.4,
            batch_size=4,
            fps=60
        )
        tracker.run(video_path)
    else:
        print("Please update the video_path or use command line arguments")
        print("Usage: python script.py --video path/to/video.mp4")