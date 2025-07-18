import torch
import cv2
import time
import numpy as np
import argparse
import os
import math
from collections import defaultdict, deque
from scipy.optimize import linear_sum_assignment
class PersonTracker:
    def __init__(self, model_name='yolov5m', confidence_threshold=0.5, device=None):
        """
        Initialize the person tracker with distance monitoring
        
        Args:
            model_name (str): YOLOv5 model variant
            confidence_threshold (float): Minimum confidence for detection
            device (str): Device to run on
        """
        self.confidence_threshold = confidence_threshold
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load YOLOv5 model
        print(f"Loading {model_name} model on {self.device}...")
        self.model = torch.hub.load('ultralytics/yolov5', model_name)
        self.model.to(self.device)
        self.model.eval() # bật chế độ đánh giá
        
        # Tracking variables
        self.tracks = {}  # {track_id: Track object} - lưu trữ thông tin của các object đang được theo dõi, các đối tượng này đã được đánh ID
        self.next_id = 1
        self.max_disappeared = 30  # Max frames before removing track
        self.max_distance = 100   # Max distance for matching detections 
        # là ngưỡng khoảng cách tối đa (tính bằng pixel) được cho phép giữa một track đang theo dõi và một detection mới để hệ thống coi đó là cùng một người (matching).
        
        # Distance monitoring parameters
        self.PERSON_HEIGHT_REAL = 1.7  # meters - chiều cao mặc định 1 người, mô phỏng
        self.SOCIAL_DISTANCE_THRESHOLD = 2.0  # meters - Khoảng cách giãn cách xã hội
        self.WARNING_DURATION = 1  # seconds - thời gian mà 2 người gần nhau tối đa trước khi cảnh báo vang lên
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0
        self.current_fps = 30  # Default FPS, will be updated from video
        
        # Distance tracking
        self.distance_history = defaultdict(lambda: deque(maxlen=3000))  # 50fps * 60s = 3000 frames/ mục đích để lưu lại khoảng cách giữa 2 người theo thời gian.
        self.warned_pairs = set()  # Track pairs that have been warned
        
        # Colors for different IDs
        self.colors = self._generate_colors(50)
        self.available_ids = deque()
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
        preds = results.pred[0] # lấy ra các dự đoán về con người - định dạng torch.tensor([N, 6]), và đương nhiên lấy ra tấm ảnh đầu tiên trong batch
        for *xyxy, conf, cls in preds: # Thứ tự các phần tử được dự đoán - xyxy conf clss
            if int(cls) == 0 and conf > self.confidence_threshold:  # Person class - lấy người và lấy những dự đoán có conf > threshold
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
        # Trả ra 1 list các dự đoán
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
    
    # nhận vào đầu ra của hàm detect_human
    # detections.append({
    #                 'bbox': (x1, y1, x2, y2),
    #                 'center': (center_x, center_y),
    #                 'confidence': float(conf),
    #                 'area': width * height,
    #                 'height_pixels': height  # Store pixel height for distance calculation
    #             })
    def update_tracks(self, detections):
        """Update tracks using Hungarian algorithm"""
        track_ids = list(self.tracks.keys()) # danh sách các keys (id) của các đối tượng đang được theo dõi
        num_tracks = len(track_ids) # số lượng đối tượng đang được theo dõi
        num_detections = len(detections) # số lượng đối tượng mới được phát hiện trong khung hình vừa rồi

        if num_tracks == 0:
            for detection in detections:
                self.tracks[self.next_id] = Track(self.next_id, detection) # khởi tạo đối tượng theo dõi mới và thêm và danh sách các đối tượng theo dõi.
                self.next_id += 1 # tăng ID lên.
            return

        # ma trận chi phí trong quá trình gán detection mới cho các tracking đã theo dõi - sử dụng 
        # trong thuật toán hungarian
        cost_matrix = np.zeros((num_tracks, num_detections), dtype=np.float32)
        for i, track_id in enumerate(track_ids): # lặp qua từng các key của các track đang được theo dõi
            track = self.tracks[track_id] # lấy ra track
            for j, detection in enumerate(detections): # lặp qua các dự đoán
                # i, j là các vị trí tương ứng trên ma trận chi phí.
                
                # khoảng cách eudiance giữa detection mới với vị trí mới nhất của đối tượng đã được theo dõi lưu trong track
                distance = self.calculate_pixel_distance(track.center, detection['center'])
                
                # tính IoU giữa detection mới với vị trí mới nhất của đối tượng đã được theo dõi lưu trong track
                iou = self.calculate_iou(track.bbox, detection['bbox'])
                cost = distance - (iou * 50) # có thể thay 50 bằng giá trị khác, tùy thuộc độ ưu tiên IoU so với distance
                cost_matrix[i, j] = cost

        # Hungarian Algorithm Input: cost_matrix có shape [num_tracks, num_detections], 
        # mỗi phần tử là chi phí gán giữa một track và một detection.
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_tracks = set() # chỉ số hàng (track index).
        assigned_detections = set() # chỉ số cột (detection index).
        # Ghép (row_ind[i], col_ind[i]) sẽ cho bạn cặp track–detection tối ưu.

        for i, j in zip(row_ind, col_ind): # lặp qua từng ô của mảng chi phí
            if cost_matrix[i, j] < self.max_distance: # loại bỏ các match có chi phí vượt ngưỡng
                track_id = track_ids[i] # lấy ra id của track đang được theo dõi
                self.tracks[track_id].update(detection=detections[j]) # cập nhật
                assigned_tracks.add(track_id) # đánh dấu các track đã match thành công rồi. thông qua id của nó
                assigned_detections.add(j) # đánh dấu các detect đã được match qua thứ tự của nó

        # Increase disappeared count for unmatched tracks - cập nhật số lượng frame liên tiếp ko được detect của các đối tượng đã được theo dõi trước đó.
        for i, track_id in enumerate(track_ids):
            if track_id not in assigned_tracks:
                self.tracks[track_id].disappeared += 1

        # Add new tracks for unmatched detections
        for j, detection in enumerate(detections):
            if j not in assigned_detections: # xem các detect nào mà chưa có track nào theo dõi, tức là đối tượng mới.
                if self.available_ids: # kiểm tra danh sách các ID sẵn sàng
                    reuse_id = self.available_ids.popleft()
                else:
                    reuse_id = self.next_id
                    self.next_id += 1
                self.tracks[reuse_id] = Track(reuse_id, detection) # thêm track mới, track vừa được khởi tạo nhằm theo dõi các đối tượng mới.

        # Remove disappeared tracks
        to_remove = [track_id for track_id, t in self.tracks.items() if t.disappeared > self.max_disappeared] # bỏ đi các track mà số lần liên tiếp đối 
        # tượng của track đó đã không xuất hiện vượt qua ngưỡng đã qua định trước
        for track_id in to_remove: # track nào bị remove thì lấy lại ID của track đó.
            del self.tracks[track_id]
            self.available_ids.append(track_id)  # Mark ID as reusable  
    
    # tính toán khoảng cách giữa các đối tượng
    def monitor_distances(self):
        """Monitor distances between all active tracks"""
        active_tracks = [(tid, track) for tid, track in self.tracks.items() if track.disappeared == 0] # Lọc ra các track mà đối tượng của nó chưa biến mất
        # tức là các track mà bbox của vật thể còn nằm trên khung hình
        
        # close_pairs.append((id1, id2, real_distance))
        close_pairs = [] # lưu lại cặp ID của 2 track, và khoảng cách 2 đối tượng trong 2 track đó, ở frame hình mới nhất của 2 track
        
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
                
                # Record distance in history - ghi lại lịch sử khoảng cách giữa cặp đối tượng.
                self.distance_history[pair_key].append(real_distance)
                
                # Check if distance is less than threshold
                if real_distance < self.SOCIAL_DISTANCE_THRESHOLD:
                    close_pairs.append((id1, id2, real_distance))
                    
                    # Check if they've been close for too long
                    if len(self.distance_history[pair_key]) > 0: # kiểm tra 1 cặp khoảng cách đã tồn tại hay chưa - nó lấy cả các cặp trong quá khứ nữa
                        # chia tb ra.
                        # Count frames where distance < threshold in recent history
                        close_frames = sum(1 for d in self.distance_history[pair_key] 
                                         if d < self.SOCIAL_DISTANCE_THRESHOLD)
                        
                        # Calculate time in seconds: tổng thời gian (giây) mà cặp đối tượng này tiếp xúc gần.
                        close_time = close_frames / self.current_fps
                        
                        # Issue warning if close for more than WARNING_DURATION
                        # warned_pairs chứa các cặp đã từng bị cảnh báo < remove khúc này bởi ID đã có cơ chế reset
                        if close_time >= self.WARNING_DURATION and pair_key not in self.warned_pairs:
                            self.warned_pairs.add(pair_key)
                            print(f"\n🚨 WARNING: Person ID {id1} and ID {id2} have been within {self.SOCIAL_DISTANCE_THRESHOLD}m for {close_time:.1f} seconds!")
                            print(f"   Current distance: {real_distance:.2f}m")
                else:
                    # Remove warning if they're no longer close
                    pair_key = (min(id1, id2), max(id1, id2))
                    self.warned_pairs.discard(pair_key) # xóa cảnh báo khi 2 người đã ko còn vi phạm khoảng cách
        
        # list tuple: close_pairs.append((id1, id2, real_distance))
        return close_pairs
    
    def draw_tracks(self, frame):
        # Truyền vào khung hình 
        """Draw tracking results on frame"""
        # Monitor distances first, tính toán khoảng cách hiện thời các đối tượng còn đang hiển thị.
        close_pairs = self.monitor_distances()
        
        # Draw tracks: vẽ các bounding box
        for track_id, track in self.tracks.items():
            if track.disappeared > 0:
                continue  # Skip disappeared tracks, bỏ qua các track mà hiện thời đối tượng ko được phát hiện.
            
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
        
        # vẽ các đường khoảng cách giữa các đối tượng vi phạm khoảng cách
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

# detection dạng
# {
#    'bbox': (x1, y1, x2, y2),
#    'center': (center_x, center_y),
#    'confidence': float(conf),
#    'area': width * height,
#    'height_pixels': height  # Store pixel height for distance calculation
# }
# khởi tạo 1 đối tượng theo dõi mới
class Track:
    def __init__(self, track_id, detection):
        self.id = track_id 
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = detection['confidence']
        self.height_pixels = detection['height_pixels']
        self.smoothed_height = detection['height_pixels']
        self.alpha = 0.6  # EMA smoothing factor: EMA - 
        # Exponential Moving Average  được dùng để làm mượt (smoothing) vị trí của các đối tượng được theo dõi qua các frame.
        
        self.disappeared = 0 # số frame liên tiếp mà đối tượng ko được phát hiện.
        self.trail = [detection['center']] # lưu vị trí các tâm đối tượng, phục vụ cho việc vẽ các đường di chuyển.
        self.max_trail_length = 20 # số lượng các đổi nối để vẽ trail

        self.age = 0 # số frame mà đối tượng này đã tồn tại, bất kể đối tượng có được nhìn thấy hay không.
        # Dùng để quyết định khi nào xóa một track quá cũ, hoặc dùng để phân tích thời gian tồn tại.
        
        self.total_visible_count = 1 # Số frame mà track này đã thực sự "thấy" được đối tượng (được match thành công với một detection)/ có thể đại diện cho mức 
        # độ tin cậy về sự xuất hiện của đối tượng này, tin cậy càng cao, tham số này càng lớn
        
        self.consecutive_invisible_count = 0 # Đếm số frame liên tiếp mà track không được match với detection nào (mất dấu).
    
    # update 1 track, Hàm này dùng để cập nhật thông tin của track mỗi khi có một detection mới khớp với track này.
    def update(self, detection):
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.confidence = detection['confidence']

        # EMA smoothing
        self.smoothed_height = self.alpha * detection['height_pixels'] + (1 - self.alpha) * self.smoothed_height # chiều cao đã được làm mượt
        self.height_pixels = self.smoothed_height # cập nhật chiều cao của đối tượng

        self.disappeared = 0 # cập nhật lại, số khung hình liên tiếp mà đối tượng biến mất là 0
        self.age += 1 # tăng tuổi của đối tượng
        self.total_visible_count += 1 # tăng tổng số khung hình có phát hiện ra đối tượng
        self.consecutive_invisible_count = 0 # cập nhật lại số lượng khung hình liên tiếp mà ko phát hiện ra đối tượng.

        # Thêm 1 điểm vào quá trình dịch chuyển của đối tượng để vẽ các đường đi
        self.trail.append(detection['center'])
        if len(self.trail) > self.max_trail_length: # xóa bỏ bớt các điểm vẽ trail cũ
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

if __name__ == "__main__":
    # Example usage
    video_path = r"D:\WorkSpace\PersonPath22\tracking-dataset\dataset\dataset1\raw_data\uid_vid_00035.mp4"
    
    if os.path.exists(video_path):
        tracker = PersonTracker(confidence_threshold=0.4)
        process_video(video_path, tracker)
    else:
        print("Please update the video_path or use command line arguments")
        print("Usage: python script.py --video path/to/video.mp4")