import torch
import cv2
import time
import numpy as np
import argparse
import os
import math
from collections import defaultdict, deque

class PersonTracker:
    def __init__(self, model_name='yolov5s', confidence_threshold=0.5, device=None):
        """
        Initialize the person tracker with distance monitoring
        
        Args:
            model_name (str): YOLOv5 model variant
            confidence_threshold (float): Minimum confidence for detection
            device (str): Device to run on
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = "cpu"
        
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
        self.distance_history = defaultdict(lambda: deque(maxlen=3000))  # 50fps * 60s = 3000 frames
        self.warned_pairs = set()  # Track pairs that have been warned
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0
        self.current_fps = 30  # Default FPS, will be updated from video
        
        # Colors for different IDs
        self.colors = self._generate_colors(50)
    
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
        # Update deque maxlen based on FPS
        max_frames = int(fps * self.WARNING_DURATION)
        self.distance_history = defaultdict(lambda: deque(maxlen=max_frames))
    
    def detect_persons(self, frame):
        """Detect persons in frame"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        start_time = time.time()
        results = self.model(img, size=640)
        inference_time = time.time() - start_time
        
        self.frame_count += 1
        self.total_time += inference_time
        
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
                    'height_pixels': height  # Store pixel height for distance calculation
                })
        
        return detections, inference_time
    
    def calculate_pixel_distance(self, center1, center2):
        """Calculate Euclidean distance between two centers in pixels"""
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_real_distance(self, center1, center2, height1_pixels, height2_pixels):
        """
        Calculate real-world distance between two persons based on their pixel heights
        
        Args:
            center1, center2: Center coordinates of the two persons
            height1_pixels, height2_pixels: Heights in pixels of the two persons
            
        Returns:
            Real distance in meters
        """
        # Average height in pixels
        avg_height_pixels = (height1_pixels + height2_pixels) / 2
        
        # Calculate pixels per meter ratio
        pixels_per_meter = avg_height_pixels / self.PERSON_HEIGHT_REAL
        
        # Calculate pixel distance
        pixel_distance = self.calculate_pixel_distance(center1, center2)
        
        # Convert to real distance
        real_distance = pixel_distance / pixels_per_meter
        
        return real_distance
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
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
        """Update tracks with new detections"""
        # If no existing tracks, create new ones
        if not self.tracks:
            for detection in detections:
                self.tracks[self.next_id] = Track(self.next_id, detection)
                self.next_id += 1
            return
        
        # Calculate cost matrix (distance + IoU)
        track_ids = list(self.tracks.keys())
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, detection in enumerate(detections):
                distance = self.calculate_pixel_distance(track.center, detection['center'])
                iou = self.calculate_iou(track.bbox, detection['bbox'])
                
                # Combined cost: distance penalty - IoU bonus
                cost = distance - (iou * 50)  # IoU bonus to favor overlapping boxes
                cost_matrix[i, j] = cost
        
        # Simple assignment using Hungarian-like approach
        matches, unmatched_tracks, unmatched_detections = self._assign_detections_to_tracks(
            cost_matrix, track_ids, detections
        )
        
        # Update matched tracks
        for track_id, detection in matches:
            self.tracks[track_id].update(detection)
        
        # Mark unmatched tracks as disappeared
        for track_id in unmatched_tracks:
            self.tracks[track_id].disappeared += 1
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            self.tracks[self.next_id] = Track(self.next_id, detection)
            self.next_id += 1
        
        # Remove tracks that have disappeared for too long
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if track.disappeared > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def _assign_detections_to_tracks(self, cost_matrix, track_ids, detections):
        """Simple assignment algorithm"""
        matches = []
        unmatched_tracks = []
        unmatched_detections = list(range(len(detections)))
        
        if cost_matrix.size == 0:
            return matches, list(range(len(track_ids))), unmatched_detections
        
        # Greedy assignment
        used_tracks = set()
        used_detections = set()
        
        # Sort by cost (ascending)
        assignments = []
        for i in range(len(track_ids)):
            for j in range(len(detections)):
                assignments.append((cost_matrix[i, j], i, j))
        
        assignments.sort()
        
        for cost, track_idx, det_idx in assignments:
            track_id = track_ids[track_idx]
            
            if track_idx in used_tracks or det_idx in used_detections:
                continue
            
            if cost < self.max_distance:  # Only assign if cost is reasonable
                matches.append((track_id, detections[det_idx]))
                used_tracks.add(track_idx)
                used_detections.add(det_idx)
        
        # Find unmatched tracks and detections
        for i, track_id in enumerate(track_ids):
            if i not in used_tracks:
                unmatched_tracks.append(track_id)
        
        unmatched_detections = [detections[i] for i in range(len(detections)) if i not in used_detections]
        
        return matches, unmatched_tracks, unmatched_detections
    
    def monitor_distances(self):
        """Monitor distances between all active tracks"""
        active_tracks = [(tid, track) for tid, track in self.tracks.items() if track.disappeared == 0]
        close_pairs = []
        
        for i, (id1, track1) in enumerate(active_tracks):
            for j, (id2, track2) in enumerate(active_tracks):
                if i >= j:  # Avoid duplicate pairs and self-comparison
                    continue
                
                # Calculate real distance
                real_distance = self.calculate_real_distance(
                    track1.center, track2.center,
                    track1.height_pixels, track2.height_pixels
                )
                
                # Create pair key (smaller ID first)
                pair_key = (min(id1, id2), max(id1, id2))
                
                # Record distance in history
                self.distance_history[pair_key].append(real_distance)
                
                # Check if distance is less than threshold
                if real_distance < self.SOCIAL_DISTANCE_THRESHOLD:
                    close_pairs.append((id1, id2, real_distance))
                    
                    # Check if they've been close for too long
                    if len(self.distance_history[pair_key]) > 0:
                        # Count frames where distance < threshold in recent history
                        close_frames = sum(1 for d in self.distance_history[pair_key] 
                                         if d < self.SOCIAL_DISTANCE_THRESHOLD)
                        
                        # Calculate time in seconds
                        close_time = close_frames / self.current_fps
                        
                        # Issue warning if close for more than WARNING_DURATION
                        if close_time >= self.WARNING_DURATION and pair_key not in self.warned_pairs:
                            self.warned_pairs.add(pair_key)
                            print(f"\n🚨 WARNING: Person ID {id1} and ID {id2} have been within {self.SOCIAL_DISTANCE_THRESHOLD}m for {close_time:.1f} seconds!")
                            print(f"   Current distance: {real_distance:.2f}m")
                else:
                    # Remove warning if they're no longer close
                    pair_key = (min(id1, id2), max(id1, id2))
                    self.warned_pairs.discard(pair_key)
        
        return close_pairs
    
    def draw_tracks(self, frame):
        """Draw tracking results on frame"""
        # Monitor distances first
        close_pairs = self.monitor_distances()
        
        # Draw tracks
        for track_id, track in self.tracks.items():
            if track.disappeared > 0:
                continue  # Skip disappeared tracks
            
            x1, y1, x2, y2 = track.bbox
            color = self.colors[track_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track trail
            if len(track.trail) > 1:
                for i in range(1, len(track.trail)):
                    cv2.line(frame, track.trail[i-1], track.trail[i], color, 2)
            
            # Draw label with ID
            label = f'ID: {track_id} ({track.confidence:.2f})'
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Background for label
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Text
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            cv2.circle(frame, track.center, 3, color, -1)
        
        # Draw distance lines for close pairs
        for id1, id2, distance in close_pairs:
            if id1 in self.tracks and id2 in self.tracks:
                track1 = self.tracks[id1]
                track2 = self.tracks[id2]
                
                # Draw line between centers
                cv2.line(frame, track1.center, track2.center, (0, 0, 255), 2)
                
                # Draw distance text
                mid_x = (track1.center[0] + track2.center[0]) // 2
                mid_y = (track1.center[1] + track2.center[1]) // 2
                
                distance_text = f'{distance:.1f}m'
                cv2.putText(frame, distance_text, (mid_x, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return close_pairs
    
    def get_track_statistics(self):
        """Get tracking statistics"""
        active_tracks = sum(1 for track in self.tracks.values() if track.disappeared == 0)
        total_tracks = len(self.tracks)
        return {
            'active_tracks': active_tracks,
            'total_tracks': total_tracks,
            'next_id': self.next_id
        }


class Track:
    def __init__(self, track_id, detection):
        """Initialize a new track"""
        self.id = track_id
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = detection['confidence']
        self.height_pixels = detection['height_pixels']
        self.disappeared = 0
        self.trail = [detection['center']]  # Track movement trail
        self.max_trail_length = 20
        
        # Additional properties
        self.age = 0
        self.total_visible_count = 1
        self.consecutive_invisible_count = 0
    
    def update(self, detection):
        """Update track with new detection"""
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = detection['confidence']
        self.height_pixels = detection['height_pixels']
        self.disappeared = 0
        self.age += 1
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0
        
        # Update trail
        self.trail.append(detection['center'])
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)


def process_video(video_path, tracker, output_path=None, show_window=True):
    """Process video for person tracking with distance monitoring"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Set FPS for accurate timing
    tracker.set_fps(fps)
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"Distance monitoring: Social distance threshold = {tracker.SOCIAL_DISTANCE_THRESHOLD}m")
    print(f"Warning threshold: {tracker.WARNING_DURATION} seconds")
    
    # Setup video writer
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_num = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Detect persons
            detections, inference_time = tracker.detect_persons(frame)
            
            # Update tracks
            tracker.update_tracks(detections)
            
            # Draw tracking results and get close pairs
            close_pairs = tracker.draw_tracks(frame)
            
            # Get statistics
            stats = tracker.get_track_statistics()
            
            # Add information overlay
            info_text = [
                f"Frame: {frame_num}/{total_frames}",
                f"FPS: {1/inference_time:.1f}",
                f"Active IDs: {stats['active_tracks']}",
                f"Total IDs: {stats['total_tracks']}",
                f"Detections: {len(detections)}",
                f"Close pairs (<{tracker.SOCIAL_DISTANCE_THRESHOLD}m): {len(close_pairs)}",
                f"Warnings issued: {len(tracker.warned_pairs)}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Add legend
            legend_y = height - 100
            legend_text = [
                "Legend:",
                "Red line: Distance < 2m",
                "Red text: Distance in meters",
                "Colored trail: Person movement"
            ]
            
            for i, text in enumerate(legend_text):
                cv2.putText(frame, text, (10, legend_y + i * 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Write frame
            if out:
                out.write(frame)
            
            # Show frame
            if show_window:
                cv2.imshow("YOLOv5 Person Tracking with Distance Monitoring", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"tracking_frame_{frame_num}.jpg", frame)
                    print(f"Saved frame {frame_num}")
                elif key == ord('r'):  # Reset tracking
                    tracker.tracks = {}
                    tracker.next_id = 1
                    tracker.distance_history.clear()
                    tracker.warned_pairs.clear()
                    print("Tracking reset!")
                elif key == ord('i'):  # Show info
                    print(f"\nCurrent Statistics:")
                    print(f"Active tracks: {stats['active_tracks']}")
                    print(f"Close pairs: {len(close_pairs)}")
                    print(f"Warnings issued: {len(tracker.warned_pairs)}")
                    for pair in tracker.warned_pairs:
                        print(f"  - ID {pair[0]} and ID {pair[1]}")
            
            # Progress indicator
            if frame_num % 100 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Active: {stats['active_tracks']}, Close pairs: {len(close_pairs)}")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        cap.release()
        if out:
            out.release()
        if show_window:
            cv2.destroyAllWindows()
        
        print(f"\nTracking complete!")
        print(f"Processed {frame_num} frames")
        print(f"Total unique persons tracked: {tracker.next_id - 1}")
        print(f"Total warnings issued: {len(tracker.warned_pairs)}")
        if tracker.warned_pairs:
            print("Warning pairs:")
            for pair in tracker.warned_pairs:
                print(f"  - ID {pair[0]} and ID {pair[1]}")


def main():
    parser = argparse.ArgumentParser(description='YOLOv5 Person Tracking with Distance Monitoring')
    parser.add_argument('--video', type=str, required=True, 
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output video file')
    parser.add_argument('--model', type=str, default='yolov5s',
                       choices=['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
                       help='YOLOv5 model variant')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to run on')
    parser.add_argument('--no-window', action='store_true',
                       help='Disable real-time window display')
    parser.add_argument('--social-distance', type=float, default=2.0,
                       help='Social distance threshold in meters (default: 2.0)')
    parser.add_argument('--warning-time', type=float, default=60.0,
                       help='Warning time threshold in seconds (default: 60.0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        return
    
    # Create tracker
    tracker = PersonTracker(
        model_name=args.model,
        confidence_threshold=args.conf,
        device=args.device
    )
    
    # Update thresholds if provided
    tracker.SOCIAL_DISTANCE_THRESHOLD = args.social_distance
    tracker.WARNING_DURATION = args.warning_time
    
    # Process video
    process_video(
        video_path=args.video,
        tracker=tracker,
        output_path=args.output,
        show_window=not args.no_window
    )

if __name__ == "__main__":
    # Example usage
    video_path = r"D:\WorkSpace\PersonPath22\tracking-dataset\dataset\dataset1\raw_data\uid_vid_00035.mp4"
    
    if os.path.exists(video_path):
        tracker = PersonTracker(confidence_threshold=0.5)
        process_video(video_path, tracker)
    else:
        print("Please update the video_path or use command line arguments")
        print("Usage: python script.py --video path/to/video.mp4")