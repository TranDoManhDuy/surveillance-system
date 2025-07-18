# tracking human

import torch
import cv2
import time
import numpy as np
import argparse
import os
import math

class PersonTracker:
    def __init__(self, model_name='yolov5s', confidence_threshold=0.5, device=None):
        """
        Initialize the person tracker
        
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
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0
        
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
                    'area': width * height
                })
        
        return detections, inference_time
    
    def calculate_distance(self, center1, center2):
        """Calculate Euclidean distance between two centers"""
        return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
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
                distance = self.calculate_distance(track.center, detection['center'])
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
    
    def draw_tracks(self, frame):
        """Draw tracking results on frame"""
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
        self.disappeared = 0
        self.age += 1
        self.total_visible_count += 1
        self.consecutive_invisible_count = 0
        
        # Update trail
        self.trail.append(detection['center'])
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)


def process_video(video_path, tracker, output_path=None, show_window=True):
    """Process video for person tracking"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
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
            
            # Draw tracking results
            tracker.draw_tracks(frame)
            
            # Get statistics
            stats = tracker.get_track_statistics()
            
            # Add information overlay
            info_text = [
                f"Frame: {frame_num}/{total_frames}",
                f"FPS: {1/inference_time:.1f}",
                f"Active IDs: {stats['active_tracks']}",
                f"Total IDs: {stats['total_tracks'] - 1}",  # -1 because next_id starts from 1
                f"Detections: {len(detections)}"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(frame, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Write frame
            if out:
                out.write(frame)
            
            # Show frame
            if show_window:
                cv2.imshow("YOLOv5 Person Tracking", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    cv2.imwrite(f"tracking_frame_{frame_num}.jpg", frame)
                    print(f"Saved frame {frame_num}")
                elif key == ord('r'):  # Reset tracking
                    tracker.tracks = {}
                    tracker.next_id = 1
                    print("Tracking reset!")
            
            # Progress indicator
            if frame_num % 100 == 0:
                progress = (frame_num / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Active tracks: {stats['active_tracks']}")
    
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


def main():
    parser = argparse.ArgumentParser(description='YOLOv5 Person Tracking')
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