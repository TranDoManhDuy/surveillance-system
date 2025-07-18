import torch
import cv2
import time
import numpy as np
import threading
import queue
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import json
from datetime import datetime
import os
import sqlite3
from pathlib import Path
import config_system
from datetime import datetime
# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{config_system.FOLDER_LOGGING}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_surveillance_system.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class FrameBatch:
    """Batch frame data để gửi qua model"""
    camera_frames: Dict[str, np.ndarray]  # camera_id -> frame
    camera_metadata: Dict[str, Dict]      # camera_id -> metadata (frame_id, timestamp, etc.)
    batch_id: int
    timestamp: float

@dataclass
class BatchResult:
    """Kết quả từ batch processing"""
    batch_id: int
    camera_results: Dict[str, List[Dict]]  # camera_id -> detections
    processing_time: float
    timestamp: float

@dataclass
class DetectionResult:
    """Kết quả detection từ một camera"""
    camera_id: str
    frame_id: int
    timestamp: float
    detections: List[Dict]
    close_pairs: List[Tuple[int, int, float]]
    frame: np.ndarray = None

@dataclass
class CameraConfig:
    """Cấu hình cho một camera"""
    camera_id: str
    source: str
    position: str
    enable_recording: bool = True
    recording_path: str = None
    confidence_threshold: float = 0.5
    social_distance_threshold: float = 2.0
    warning_duration: float = 1.0
    loop_video: bool = True  # Thêm option để loop video

class BatchProcessor:
    """Xử lý batch frames từ nhiều camera"""
    
    def __init__(self, batch_size: int = 8, max_wait_time: float = 0.05):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5m')
        self.model.to(self.device)
        self.model.eval()
        
        # Batch processing
        self.input_queue = queue.Queue(maxsize=100)
        self.output_queue = queue.Queue(maxsize=100)
        self.batch_id_counter = 0
        self.running = False
        
        # Performance tracking
        self.batch_times = deque(maxlen=100)
        self.logger = logging.getLogger("BatchProcessor")
        
        # Start processing thread
        self.processor_thread = threading.Thread(target=self._batch_processing_loop)
        self.processor_thread.daemon = True
    
    def start(self):
        """Bắt đầu batch processor"""
        self.running = True
        self.processor_thread.start()
        self.logger.info(f"BatchProcessor started with batch_size={self.batch_size}")
    
    def stop(self):
        """Dừng batch processor"""
        self.running = False
        if self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5.0)
        self.logger.info("BatchProcessor stopped")
    
    def add_frame(self, camera_id: str, frame: np.ndarray, metadata: Dict):
        """Thêm frame vào batch queue"""
        try:
            self.input_queue.put((camera_id, frame, metadata), timeout=0.01)
        except queue.Full:
            self.logger.warning("Batch input queue full, dropping frame")
    
    def get_results(self) -> Optional[BatchResult]:
        """Lấy kết quả batch processing"""
        try:
            return self.output_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _batch_processing_loop(self):
        """Main batch processing loop"""
        pending_frames = {}  # camera_id -> (frame, metadata)
        last_batch_time = time.time()
        
        while self.running:
            try:
                # Thu thập frames cho batch
                while len(pending_frames) < self.batch_size:
                    try:
                        camera_id, frame, metadata = self.input_queue.get(timeout=0.01)
                        pending_frames[camera_id] = (frame, metadata)
                    except queue.Empty:
                        break
                
                # Kiểm tra điều kiện để xử lý batch
                current_time = time.time()
                should_process = (
                    len(pending_frames) >= self.batch_size or
                    (len(pending_frames) > 0 and (current_time - last_batch_time) >= self.max_wait_time)
                )
                
                if should_process and pending_frames:
                    # Tạo batch
                    batch = self._create_batch(pending_frames)
                    
                    # Xử lý batch
                    result = self._process_batch(batch)
                    
                    # Gửi kết quả
                    try:
                        self.output_queue.put(result, timeout=0.01)
                    except queue.Full:
                        self.logger.warning("Batch output queue full")
                    
                    # Reset
                    pending_frames.clear()
                    last_batch_time = current_time
                else:
                    time.sleep(0.001)  # Ngủ ngắn để tránh busy waiting
                    
            except Exception as e:
                self.logger.error(f"Error in batch processing loop: {e}")
                time.sleep(0.01)
    
    def _create_batch(self, pending_frames: Dict) -> FrameBatch:
        """Tạo batch từ pending frames"""
        camera_frames = {}
        camera_metadata = {}
        
        for camera_id, (frame, metadata) in pending_frames.items():
            camera_frames[camera_id] = frame
            camera_metadata[camera_id] = metadata
        
        batch_id = self.batch_id_counter
        self.batch_id_counter += 1
        
        return FrameBatch(
            camera_frames=camera_frames,
            camera_metadata=camera_metadata,
            batch_id=batch_id,
            timestamp=time.time()
        )
    
    def _process_batch(self, batch: FrameBatch) -> BatchResult:
        """Xử lý batch frames qua model"""
        start_time = time.time()
        
        # Chuẩn bị input cho model
        batch_images = []
        camera_order = []
        
        for camera_id, frame in batch.camera_frames.items():
            # Chuyển đổi BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_images.append(rgb_frame)
            camera_order.append(camera_id)
        
        # Chạy model trên batch
        with torch.no_grad():
            results = self.model(batch_images, size=640)
        
        # Xử lý kết quả
        camera_results = {}
        for i, camera_id in enumerate(camera_order):
            detections = self._extract_detections(
                results.pred[i], 
                batch.camera_metadata[camera_id].get('confidence_threshold', 0.5)
            )
            camera_results[camera_id] = detections
        
        processing_time = time.time() - start_time
        self.batch_times.append(processing_time)
        
        # Log performance
        if len(self.batch_times) % 50 == 0:
            avg_time = sum(self.batch_times) / len(self.batch_times)
            self.logger.info(f"Batch processing avg time: {avg_time*1000:.2f}ms, "
                           f"batch_size: {len(batch_images)}")
        
        return BatchResult(
            batch_id=batch.batch_id,
            camera_results=camera_results,
            processing_time=processing_time,
            timestamp=time.time()
        )
    
    def _extract_detections(self, predictions, confidence_threshold: float) -> List[Dict]:
        """Trích xuất detections từ predictions"""
        detections = []
        
        for *xyxy, conf, cls in predictions:
            if int(cls) == 0 and conf > confidence_threshold:  # person class
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
        
        return detections

class DatabaseManager:
    """Quản lý database để lưu trữ kết quả"""
    def __init__(self, db_path: str = "surveillance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Khởi tạo database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bảng events
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                event_type TEXT,
                timestamp REAL,
                person_id1 INTEGER,
                person_id2 INTEGER,
                distance REAL,
                description TEXT
            )
        ''')
        
        # Bảng statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                timestamp REAL,
                total_persons INTEGER,
                active_persons INTEGER,
                violations INTEGER
            )
        ''')
        
        # Bảng performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                batch_size INTEGER,
                processing_time REAL,
                fps REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_event(self, camera_id: str, event_type: str, person_id1: int, 
                  person_id2: int = None, distance: float = None, description: str = ""):
        """Ghi lại sự kiện"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO events (camera_id, event_type, timestamp, person_id1, person_id2, distance, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (camera_id, event_type, time.time(), person_id1, person_id2, distance, description))
        
        conn.commit()
        conn.close()
    
    def log_statistics(self, camera_id: str, total_persons: int, active_persons: int, violations: int):
        """Ghi lại thống kê"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO statistics (camera_id, timestamp, total_persons, active_persons, violations)
            VALUES (?, ?, ?, ?, ?)
        ''', (camera_id, time.time(), total_persons, active_persons, violations))
        
        conn.commit()
        conn.close()
    
    def log_performance(self, batch_size: int, processing_time: float, fps: float):
        """Ghi lại performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance (timestamp, batch_size, processing_time, fps)
            VALUES (?, ?, ?, ?)
        ''', (time.time(), batch_size, processing_time, fps))
        
        conn.commit()
        conn.close()

class PersonTracker:
    """Enhanced PersonTracker - chỉ xử lý tracking, không detection"""
    def __init__(self, camera_id: str, config: CameraConfig):
        self.camera_id = camera_id
        self.config = config
        
        # Tracking variables
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = 30
        self.max_distance = 100
        
        # Distance monitoring
        self.PERSON_HEIGHT_REAL = 1.7
        self.SOCIAL_DISTANCE_THRESHOLD = config.social_distance_threshold
        self.WARNING_DURATION = config.warning_duration
        
        # Performance tracking
        self.frame_count = 0
        self.current_fps = 30
        self.distance_history = defaultdict(lambda: deque(maxlen=90))
        self.warned_pairs = set()
        
        # Colors for visualization
        self.colors = self._generate_colors(50)
        
        # Logger
        self.logger = logging.getLogger(f"Tracker-{camera_id}")
    
    def _generate_colors(self, n):
        """Generate distinct colors"""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = tuple(int(c * 255) for c in self._hsv_to_rgb(hue, 0.8, 0.9))
            colors.append(rgb)
        return colors
    
    def _hsv_to_rgb(self, h, s, v):
        import colorsys
        return colorsys.hsv_to_rgb(h, s, v)
    
    def calculate_real_distance(self, center1, center2, height1, height2):
        """Calculate real distance between two persons"""
        avg_height = (height1 + height2) / 2
        pixels_per_meter = avg_height / self.PERSON_HEIGHT_REAL
        
        pixel_distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return pixel_distance / pixels_per_meter
    
    def update_tracks(self, detections):
        """Update tracks using Hungarian algorithm"""
        if not self.tracks:
            for detection in detections:
                self.tracks[self.next_id] = Track(self.next_id, detection)
                self.next_id += 1
            return
        
        # Hungarian algorithm implementation
        track_ids = list(self.tracks.keys())
        if not track_ids or not detections:
            return
        
        # Calculate cost matrix
        cost_matrix = np.zeros((len(track_ids), len(detections)))
        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for j, detection in enumerate(detections):
                distance = np.sqrt((track.center[0] - detection['center'][0])**2 + 
                                 (track.center[1] - detection['center'][1])**2)
                cost_matrix[i, j] = distance
        
        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        assigned_tracks = set()
        assigned_detections = set()
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.max_distance:
                track_id = track_ids[i]
                self.tracks[track_id].update(detections[j])
                assigned_tracks.add(track_id)
                assigned_detections.add(j)
        
        # Update disappeared tracks
        for track_id in track_ids:
            if track_id not in assigned_tracks:
                self.tracks[track_id].disappeared += 1
        
        # Add new tracks
        for j, detection in enumerate(detections):
            if j not in assigned_detections:
                self.tracks[self.next_id] = Track(self.next_id, detection)
                self.next_id += 1
        
        # Remove disappeared tracks
        to_remove = [tid for tid, t in self.tracks.items() if t.disappeared > self.max_disappeared]
        for tid in to_remove:
            del self.tracks[tid]
    
    def monitor_distances(self):
        """Monitor distances between persons"""
        active_tracks = [(tid, track) for tid, track in self.tracks.items() if track.disappeared == 0]
        close_pairs = []
        
        for i, (id1, track1) in enumerate(active_tracks):
            for j, (id2, track2) in enumerate(active_tracks):
                if i >= j:
                    continue
                
                distance = self.calculate_real_distance(
                    track1.center, track2.center,
                    track1.height_pixels, track2.height_pixels
                )
                
                pair_key = (min(id1, id2), max(id1, id2))
                self.distance_history[pair_key].append(distance)
                
                if distance < self.SOCIAL_DISTANCE_THRESHOLD:
                    close_pairs.append((id1, id2, distance))
                    
                    # Check warning condition
                    if len(self.distance_history[pair_key]) > 0:
                        close_frames = sum(1 for d in self.distance_history[pair_key] 
                                         if d < self.SOCIAL_DISTANCE_THRESHOLD)
                        close_time = close_frames / self.current_fps
                        
                        if close_time >= self.WARNING_DURATION and pair_key not in self.warned_pairs:
                            self.warned_pairs.add(pair_key)
                            self.logger.warning(f"Social distance violation: ID {id1} and {id2} "
                                              f"too close for {close_time:.1f}s (distance: {distance:.2f}m)")
                else:
                    self.warned_pairs.discard(pair_key)
        
        return close_pairs
    
    def draw_tracks(self, frame):
        """Draw tracking results"""
        close_pairs = self.monitor_distances()
        
        # Draw tracks
        for track_id, track in self.tracks.items():
            if track.disappeared > 0:
                continue
            
            x1, y1, x2, y2 = track.bbox
            color = self.colors[track_id % len(self.colors)]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID label
            label = f'ID: {track_id}'
            cv2.putText(frame, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw distance violations
        for id1, id2, distance in close_pairs:
            if id1 in self.tracks and id2 in self.tracks:
                track1 = self.tracks[id1]
                track2 = self.tracks[id2]
                
                cv2.line(frame, track1.center, track2.center, (0, 0, 255), 2)
                
                mid_x = (track1.center[0] + track2.center[0]) // 2
                mid_y = (track1.center[1] + track2.center[1]) // 2
                cv2.putText(frame, f'{distance:.1f}m', (mid_x, mid_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return close_pairs
    
    def get_statistics(self):
        """Get current statistics"""
        active_tracks = sum(1 for track in self.tracks.values() if track.disappeared == 0)
        return {
            'active_tracks': active_tracks,
            'total_tracks': len(self.tracks),
            'violations': len(self.warned_pairs)
        }

class Track:
    """Individual track object"""
    def __init__(self, track_id, detection):
        self.id = track_id
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = detection['confidence']
        self.height_pixels = detection['height_pixels']
        self.disappeared = 0
        self.trail = [detection['center']]
        self.max_trail_length = 30
    
    def update(self, detection):
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = detection['confidence']
        self.height_pixels = detection['height_pixels']
        self.disappeared = 0
        
        self.trail.append(detection['center'])
        if len(self.trail) > self.max_trail_length:
            self.trail.pop(0)

class ImprovedCameraWorker(threading.Thread):
    """Improved camera worker with better video handling"""
    def __init__(self, config: CameraConfig, batch_processor: BatchProcessor, 
                 db_manager: DatabaseManager):
        super().__init__()
        self.config = config
        self.batch_processor = batch_processor
        self.db_manager = db_manager
        self.running = False
        self.tracker = PersonTracker(config.camera_id, config)
        self.logger = logging.getLogger(f"Camera-{config.camera_id}")
        
        # Video capture
        self.cap = None
        self.frame_count = 0
        self.total_frames = 0
        self.is_video_file = isinstance(config.source, str) and not config.source.isdigit()
        
        # Recording
        self.video_writer = None
        if config.enable_recording and config.recording_path:
            os.makedirs(config.recording_path, exist_ok=True)
        
        # Latest frame storage
        self.latest_frame = None
        self.latest_result = None
        self.latest_frame_lock = threading.Lock()
        
        # FPS control
        self.target_fps = 30
        self.frame_time = 1.0 / self.target_fps
        self.last_frame_time = 0
        
        # Status
        self.status = "Initializing"
        self.is_active = True
    
    def run(self):
        """Main processing loop"""
        self.running = True
        self.logger.info(f"Starting camera {self.config.camera_id}")
        
        while self.running and (self.config.loop_video or self.frame_count == 0):
            try:
                # Open/reopen video source
                if self.cap is None or (self.is_video_file and self.frame_count >= self.total_frames):
                    self._open_video_source()
                    if not self.cap or not self.cap.isOpened():
                        self.logger.error(f"Cannot open camera source: {self.config.source}")
                        self.status = "Error: Cannot open source"
                        self.is_active = False
                        break
                
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    if self.is_video_file:
                        if self.config.loop_video:
                            self.logger.info(f"Video ended for {self.config.camera_id}, restarting...")
                            self.cap.release()
                            self.cap = None
                            self.frame_count = 0
                            continue
                        else:
                            self.logger.info(f"Video ended for {self.config.camera_id}")
                            self.status = "Video ended"
                            self.is_active = False
                            break
                    else:
                        self.logger.warning(f"Failed to read frame from {self.config.camera_id}")
                        time.sleep(0.1)
                        continue
                
                # FPS control
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)
                self.last_frame_time = time.time()
                
                self.frame_count += 1
                self.status = f"Active (Frame: {self.frame_count})"
                
                # Store latest frame
                with self.latest_frame_lock:
                    self.latest_frame = frame.copy()
                
                # Send frame to batch processor
                metadata = {
                    'frame_id': self.frame_count,
                    'timestamp': time.time(),
                    'confidence_threshold': self.config.confidence_threshold
                }
                
                self.batch_processor.add_frame(self.config.camera_id, frame, metadata)
                
                # Record video (original frame)
                if self.video_writer:
                    self.video_writer.write(frame)
                    
            except Exception as e:
                self.logger.error(f"Error in camera worker: {e}")
                self.status = f"Error: {str(e)}"
                time.sleep(0.1)
        
        self.cleanup()
    
    def _open_video_source(self):
        """Open or reopen video source"""
        try:
            # Convert source to int if it's a digit string
            source = self.config.source
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            
            self.cap = cv2.VideoCapture(source)
            
            if self.cap.isOpened():
                # Get video properties
                fps = int(self.cap.get(cv2.CAP_PROP_FPS))
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if self.is_video_file:
                    self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.logger.info(f"Opened video file: {self.config.source}, "
                                   f"Total frames: {self.total_frames}, FPS: {fps}")
                
                self.tracker.current_fps = max(fps, 1)
                self.target_fps = min(fps, 30)  # Cap at 30 FPS
                self.frame_time = 1.0 / self.target_fps
                
                # Setup video writer for first time
                if self.video_writer is None and self.config.enable_recording:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(self.config.recording_path, 
                                             f"{self.config.camera_id}_{timestamp}.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
        except Exception as e:
            self.logger.error(f"Error opening video source: {e}")
            self.cap = None
    
    def get_latest_frame(self):
        """Get the latest frame with status overlay"""
        with self.latest_frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
                
                # Add status overlay
                cv2.putText(frame, f"Status: {self.status}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                return frame
            else:
                # Create blank frame with status
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"{self.config.camera_id}: {self.status}", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return frame
    
    def get_latest_result(self):
        """Get the latest processed result"""
        with self.latest_frame_lock:
            return self.latest_result
    
    def process_detections(self, detections: List[Dict], frame: np.ndarray) -> DetectionResult:
        """Process detection results from batch processor"""
        # Update tracks
        self.tracker.update_tracks(detections)
        
        # Draw tracks and monitor distances
        close_pairs = self.tracker.draw_tracks(frame)
        
        # Create result
        result = DetectionResult(
            camera_id=self.config.camera_id,
            frame_id=self.frame_count,
            timestamp=time.time(),
            detections=detections,
            close_pairs=close_pairs,
            frame=frame
        )
        
        # Store latest result
        with self.latest_frame_lock:
            self.latest_result = result
        
        # Log statistics periodically
        if self.frame_count % 300 == 0:
            stats = self.tracker.get_statistics()
            self.db_manager.log_statistics(
                self.config.camera_id,
                stats['total_tracks'],
                stats['active_tracks'],
                stats['violations']
            )
        
        # Log violations
        for id1, id2, distance in close_pairs:
            pair_key = (min(id1, id2), max(id1, id2))
            if pair_key in self.tracker.warned_pairs:
                self.db_manager.log_event(
                    self.config.camera_id,
                    "social_distance_violation",
                    id1, id2, distance,
                    f"Distance: {distance:.2f}m"
                )
        
        return result
    
    def stop(self):
        """Stop the camera worker"""
        self.running = False
        self.logger.info(f"Stopping camera {self.config.camera_id}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.cap:
                self.cap.release()
            if self.video_writer:
                self.video_writer.release()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

class MultiCameraSurveillanceSystem:
    """Main surveillance system with improved real-time display"""
    def __init__(self, config_file: str = "cameras.json", batch_size: int = 8):
        self.config_file = config_file
        self.batch_size = batch_size
        self.cameras = {}
        self.camera_workers = {}
        self.db_manager = DatabaseManager()
        self.running = False
        self.logger = logging.getLogger("SurveillanceSystem")
        
        # Batch processor
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        
        # Results processing
        self.result_queue = queue.Queue(maxsize=100)
        
        # Load configuration
        self.load_config()
        
        # Display management
        self.display_windows = {}
        self.grid_size = self._calculate_grid_size()
        
        # Frame caching for real-time display
        self.frame_cache = {}
        self.frame_cache_lock = threading.Lock()
        
        # Performance metrics
        self.fps_tracker = defaultdict(lambda: deque(maxlen=30))
        self.last_update_time = defaultdict(float)
    
    def load_config(self):
        """Load camera configuration from JSON file"""
        default_config = {
            "cameras": [
                {
                    "camera_id": "CAM001",
                    "source": r"D:\WorkSpace\PersonPath22\tracking-dataset\dataset\dataset1\raw_data\uid_vid_00000.mp4",
                    "position": "Entrance",
                    "enable_recording": True,
                    "recording_path": "./recordings",
                    "confidence_threshold": 0.4,
                    "social_distance_threshold": 2.0,
                    "warning_duration": 1.0,
                    "loop_video": True
                },
                {
                    "camera_id": "CAM002",
                    "source": "0",
                    "position": "Main Hall",
                    "enable_recording": True,
                    "recording_path": "./recordings",
                    "confidence_threshold": 0.4,
                    "social_distance_threshold": 2.0,
                    "warning_duration": 1.0,
                    "loop_video": False
                }
            ]
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = default_config
                with open(self.config_file, 'w') as f:
                    json.dump(config, f, indent=2)
            
            for cam_config in config['cameras']:
                camera_config = CameraConfig(**cam_config)
                self.cameras[camera_config.camera_id] = camera_config
                
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            for cam_config in default_config['cameras']:
                camera_config = CameraConfig(**cam_config)
                self.cameras[camera_config.camera_id] = camera_config
    
    def _calculate_grid_size(self):
        """Calculate grid size for display"""
        num_cameras = len(self.cameras)
        if num_cameras <= 1:
            return (1, 1)
        elif num_cameras <= 4:
            return (2, 2)
        elif num_cameras <= 9:
            return (3, 3)
        else:
            return (4, 4)
    
    def start(self):
        """Start the surveillance system"""
        self.logger.info("Starting Multi-Camera Surveillance System with Batch Processing")
        self.running = True
        
        # Start batch processor
        self.batch_processor.start()
        
        # Start camera workers
        for camera_id, config in self.cameras.items():
            worker = ImprovedCameraWorker(config, self.batch_processor, self.db_manager)
            worker.start()
            self.camera_workers[camera_id] = worker
        
        # Start processing threads
        threads = [
            threading.Thread(target=self._process_batch_results, daemon=True),
            threading.Thread(target=self._update_display_cache, daemon=True),
            threading.Thread(target=self._display_loop, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(0.1)
                
                # Check if all camera workers are still active
                active_workers = sum(1 for w in self.camera_workers.values() if w.is_active)
                if active_workers == 0:
                    self.logger.info("All camera workers have stopped")
                    break
                    
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.stop()
    
    def _process_batch_results(self):
        """Process results from batch processor"""
        while self.running:
            try:
                batch_result = self.batch_processor.get_results()
                if batch_result is None:
                    time.sleep(0.001)
                    continue
                
                # Process results for each camera
                for camera_id, detections in batch_result.camera_results.items():
                    if camera_id in self.camera_workers:
                        worker = self.camera_workers[camera_id]
                        
                        # Get frame from worker
                        frame = worker.get_latest_frame()
                        if frame is not None:
                            # Process detections
                            result = worker.process_detections(detections, frame.copy())
                            
                            # Update display cache
                            with self.frame_cache_lock:
                                self.frame_cache[camera_id] = result.frame
                                self.last_update_time[camera_id] = time.time()
                
                # Log performance
                self.db_manager.log_performance(
                    len(batch_result.camera_results),
                    batch_result.processing_time,
                    len(batch_result.camera_results) / batch_result.processing_time
                )
                
            except Exception as e:
                self.logger.error(f"Error processing batch results: {e}")
                time.sleep(0.01)
    
    def _update_display_cache(self):
        """Update display cache with latest frames"""
        while self.running:
            try:
                current_time = time.time()
                
                for camera_id, worker in self.camera_workers.items():
                    # Get latest frame
                    frame = worker.get_latest_frame()
                    
                    if frame is not None:
                        # Calculate FPS
                        if camera_id in self.last_update_time:
                            fps = 1.0 / (current_time - self.last_update_time[camera_id])
                            self.fps_tracker[camera_id].append(fps)
                        
                        # Update display if no recent processed frame
                        with self.frame_cache_lock:
                            if camera_id not in self.frame_cache or \
                               (current_time - self.last_update_time.get(camera_id, 0)) > 0.1:
                                self.frame_cache[camera_id] = frame
                                self.last_update_time[camera_id] = current_time
                
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                self.logger.error(f"Error updating display cache: {e}")
                time.sleep(0.1)
    
    def _display_loop(self):
        """Display loop for showing all camera feeds"""
        cv2.namedWindow("Multi-Camera Surveillance System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Multi-Camera Surveillance System", 1280, 720)
        
        while self.running:
            try:
                # Create grid display
                grid_rows, grid_cols = self.grid_size
                display_width = 1280
                display_height = 720
                
                cell_width = display_width // grid_cols
                cell_height = display_height // grid_rows
                
                display_frame = np.zeros((display_height, display_width, 3), dtype=np.uint8)
                
                # Draw camera feeds
                with self.frame_cache_lock:
                    camera_ids = list(self.frame_cache.keys())
                    
                    for i, camera_id in enumerate(camera_ids[:grid_rows * grid_cols]):
                        if camera_id in self.frame_cache:
                            frame = self.frame_cache[camera_id]
                            
                            # Resize frame to fit grid cell
                            resized = cv2.resize(frame, (cell_width - 2, cell_height - 2))
                            
                            # Calculate position in grid
                            row = i // grid_cols
                            col = i % grid_cols
                            
                            y1 = row * cell_height + 1
                            y2 = (row + 1) * cell_height - 1
                            x1 = col * cell_width + 1
                            x2 = (col + 1) * cell_width - 1
                            
                            display_frame[y1:y2, x1:x2] = resized
                            
                            # Add camera info overlay
                            info_bg = display_frame[y1:y1+40, x1:x2].copy()
                            cv2.addWeighted(info_bg, 0.7, np.zeros_like(info_bg), 0.3, 0, info_bg)
                            display_frame[y1:y1+40, x1:x2] = info_bg
                            
                            # Camera ID
                            cv2.putText(display_frame, camera_id, (x1 + 10, y1 + 25),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # FPS
                            if camera_id in self.fps_tracker and self.fps_tracker[camera_id]:
                                avg_fps = sum(self.fps_tracker[camera_id]) / len(self.fps_tracker[camera_id])
                                cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", 
                                           (x2 - 100, y1 + 25),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            
                            # Worker status
                            if camera_id in self.camera_workers:
                                worker = self.camera_workers[camera_id]
                                status_color = (0, 255, 0) if worker.is_active else (0, 0, 255)
                                cv2.circle(display_frame, (x2 - 15, y1 + 20), 8, status_color, -1)
                
                # Add system info
                self._add_system_info(display_frame)
                
                # Display
                cv2.imshow("Multi-Camera Surveillance System", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"surveillance_screenshot_{timestamp}.jpg", display_frame)
                    self.logger.info(f"Screenshot saved")
                elif key == ord('p'):
                    # Print performance stats
                    self._print_performance_stats()
                elif key == ord('r'):
                    # Reset trackers
                    for worker in self.camera_workers.values():
                        worker.tracker.tracks.clear()
                        worker.tracker.next_id = 1
                    self.logger.info("All trackers reset")
                
            except Exception as e:
                self.logger.error(f"Error in display loop: {e}")
                time.sleep(0.1)
        
        cv2.destroyAllWindows()
    
    def _add_system_info(self, display_frame):
        """Add system information overlay"""
        height, width = display_frame.shape[:2]
        
        # Create info panel
        info_height = 60
        info_bg = np.zeros((info_height, width, 3), dtype=np.uint8)
        display_frame[height-info_height:height, :] = info_bg
        
        # System time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(display_frame, current_time, (10, height - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Performance info
        if hasattr(self.batch_processor, 'batch_times') and self.batch_processor.batch_times:
            avg_time = sum(self.batch_processor.batch_times) / len(self.batch_processor.batch_times)
            throughput = self.batch_size / avg_time
            
            perf_text = f"Batch: {self.batch_size} | Avg Time: {avg_time*1000:.1f}ms | Throughput: {throughput:.1f} fps"
            cv2.putText(display_frame, perf_text, (10, height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Active cameras
        active_count = sum(1 for w in self.camera_workers.values() if w.is_active)
        cv2.putText(display_frame, f"Active Cameras: {active_count}/{len(self.camera_workers)}", 
                   (width - 200, height - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Controls help
        help_text = "Q: Quit | S: Screenshot | P: Performance | R: Reset Trackers"
        cv2.putText(display_frame, help_text, (width - 500, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _print_performance_stats(self):
        """Print performance statistics"""
        if hasattr(self.batch_processor, 'batch_times') and self.batch_processor.batch_times:
            times = list(self.batch_processor.batch_times)
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print("\n" + "="*60)
            print("PERFORMANCE STATISTICS")
            print("="*60)
            print(f"Batch Size: {self.batch_size}")
            print(f"GPU Device: {self.batch_processor.device}")
            print(f"Avg Batch Time: {avg_time*1000:.2f}ms")
            print(f"Min Batch Time: {min_time*1000:.2f}ms")
            print(f"Max Batch Time: {max_time*1000:.2f}ms")
            print(f"Estimated FPS: {1/avg_time:.1f}")
            print(f"Throughput: {self.batch_size/avg_time:.1f} frames/sec")
            
            print("\nCamera Statistics:")
            for camera_id, worker in self.camera_workers.items():
                stats = worker.tracker.get_statistics()
                print(f"  {camera_id}: Active={stats['active_tracks']}, "
                      f"Total={stats['total_tracks']}, Violations={stats['violations']}")
            print("="*60)
    
    def stop(self):
        """Stop the surveillance system"""
        self.logger.info("Stopping surveillance system")
        self.running = False
        
        # Stop batch processor
        self.batch_processor.stop()
        
        # Stop all camera workers
        for worker in self.camera_workers.values():
            worker.stop()
        
        # Wait for workers to finish
        for worker in self.camera_workers.values():
            worker.join(timeout=5.0)
        
        cv2.destroyAllWindows()
        self.logger.info("Surveillance system stopped")

def main():
    """Main function with improved configuration"""
    print("Multi-Camera Surveillance System with Real-time Display")
    print("="*60)
    
    # Configuration
    camera_sources = [
        config_system.CAMERA_ID_1,
        config_system.CAMERA_ID_2,
        config_system.CAMERA_ID_3,
        config_system.CAMERA_ID_4
    ]
    
    recommended_batch_size = 4
    # Display GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        recommended_batch_size = 8 if gpu_memory >= 4 else 4
    else:
        print("GPU: Not available - using CPU")
        recommended_batch_size = 2
    
    # Create configuration
    config = {
        "cameras": []
    }
    
    for i, source in enumerate(camera_sources):
        is_video_file = isinstance(source, str) and not source.isdigit()
        camera_config = {
            "camera_id": f"CAM{i+1:03d}",
            "source": source,
            "position": f"Position_{i+1}",
            "enable_recording": config_system.ENABLE_RECORDING,
            "recording_path": config_system.RECORDING_PATH,
            "confidence_threshold": config_system.CONFIDENCE_THRESHOLD,
            "social_distance_threshold": config_system.SOCIAL_DISTANCE_THRESHOLD,
            "warning_duration": config_system.WARNING_DURATION,
            "loop_video": is_video_file  # Loop video files, not webcam
        }
        config["cameras"].append(camera_config)
    
    # Save configuration
    with open("cameras.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration:")
    print(f"  Cameras: {len(camera_sources)}")
    print(f"  Batch Size: {recommended_batch_size}")
    print(f"  Loop Videos: Yes")
    print("\nControls:")
    print("  Q - Quit")
    print("  S - Save screenshot")
    print("  P - Print performance stats")
    print("  R - Reset all trackers")
    print("="*60)
    try:
        # Start system
        system = MultiCameraSurveillanceSystem(batch_size=recommended_batch_size)
        system.start()
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()