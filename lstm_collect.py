import os
import csv
import copy
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque

# Optional: suppress verbose TensorFlow and MediaPipe logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# =====================================================
# CONFIGURATION
# =====================================================
VIDEO_ROOT = "C:\\Users\\Dell\\Downloads\\LSTM"  # Folder containing video files
OUTPUT_CSV = "model/fsl_words_classifier/fsl_words.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Data collection parameters
TARGET_HAND = "right"  # "right" or "left" - which hand to track
MAX_FRAMES_PER_VIDEO = 16  # Maximum frames to collect per video (matches TIME_STEPS)
FRAME_INTERVAL = 4  # Collect every Nth frame (1 = every frame, 2 = every other frame)
SKIP_INITIAL_FRAMES = 10  # Skip this many frames after hand is first detected
HISTORY_LENGTH = 16  # Number of coordinate points to track (must match app.py)

# Finger tracking per class (landmark indices)
# 4=thumb_tip, 8=index_tip, 12=middle_tip, 16=ring_tip, 20=pinky_tip
CLASS_FINGER_MAP = {
    0: 20,  # Class 0 tracks pinky fingertip
    1: 8,   # Class 1 tracks index fingertip
    # Add more classes as needed, default will be index (8)
}

# MediaPipe configuration
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Video file extensions to process
VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']

# =====================================================
# INITIALIZE MediaPipe Hands
# =====================================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # False for video processing
    max_num_hands=1,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE
)

# =====================================================
# UTILITY FUNCTIONS (from app.py)
# =====================================================

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    
    return landmark_point


def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = copy.deepcopy(point_history)
    
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]
        
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    
    # Convert to a one-dimensional list
    temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
    
    return temp_point_history


def draw_point_history(image, point_history):
    """Visualize the point history on the image"""
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                     (152, 251, 152), 2)
    
    pts = [tuple(p) for p in point_history if p[0] != 0 or p[1] != 0]
    for i in range(1, len(pts)):
        pt1, pt2 = pts[i-1], pts[i]
        cv.line(image, pt1, pt2, (0, 0, 0), 4)
        cv.line(image, pt1, pt2, (152, 251, 152), 2)
    
    return image


def flip_landmarks_horizontally(landmark_list, image_width):
    """Flip landmark coordinates horizontally"""
    flipped_landmarks = []
    for point in landmark_list:
        flipped_x = image_width - 1 - point[0]
        flipped_landmarks.append([flipped_x, point[1]])
    return flipped_landmarks


def should_flip_hand(detected_hand_label, target_hand):
    """Determine if hand should be flipped based on target hand preference"""
    if target_hand == "right" and detected_hand_label == "Left":
        return True
    elif target_hand == "left" and detected_hand_label == "Right":
        return True
    return False


# =====================================================
# VIDEO PROCESSING FUNCTIONS
# =====================================================

def get_video_files(root_dir):
    """Get all video files organized by class (folder name)"""
    video_files = {}
    
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        videos = []
        for file in sorted(os.listdir(class_path)):
            if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                videos.append(os.path.join(class_path, file))
        
        if videos:
            video_files[class_name] = videos
    
    return video_files


def process_video(video_path, class_id, class_name, video_index, total_videos, csv_writer):
    """Process a single video and collect point history data"""
    cap = cv.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"  ✗ Could not open video: {video_path}")
        return 0
    
    # Get video properties
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    
    print(f"  [{video_index+1}/{total_videos}] Processing: {os.path.basename(video_path)}")
    print(f"      Total frames: {total_frames}, FPS: {fps:.2f}")
    
    # Initialize tracking variables
    point_history = deque(maxlen=HISTORY_LENGTH)
    frame_count = 0
    collected_count = 0
    hand_detected = False
    frames_since_detection = 0
    
    while cap.isOpened() and collected_count < MAX_FRAMES_PER_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Skip frames based on FRAME_INTERVAL
        if (frame_count - 1) % FRAME_INTERVAL != 0:
            continue
        
        # Fix rotation - rotate 90 degrees counter-clockwise to correct orientation
        frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
        
        # Do NOT flip for display - keep original orientation
        debug_image = copy.deepcopy(frame)
        
        # Convert to RGB for MediaPipe
        image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        # Process hand detection
        if results.multi_hand_landmarks is not None:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0] if results.multi_handedness else None
            detected_hand_label = handedness.classification[0].label if handedness else "Unknown"
            
            # Get landmark list
            landmark_list = calc_landmark_list(frame, hand_landmarks)
            
            # Check if we need to flip the hand
            if should_flip_hand(detected_hand_label, TARGET_HAND):
                landmark_list = flip_landmarks_horizontally(landmark_list, frame.shape[1])
                detected_hand_label = TARGET_HAND
            
            # Mark that hand has been detected
            if not hand_detected:
                hand_detected = True
                frames_since_detection = 0
                print(f"      Hand detected at frame {frame_count}")
            
            frames_since_detection += 1
            
            # Determine which finger to track based on class_id
            finger_landmark = CLASS_FINGER_MAP.get(class_id, 8)  # Default to index finger (8)
            
            # Track the specified fingertip
            point_history.append(landmark_list[finger_landmark])
            
            # Only start collecting after skipping initial frames
            if frames_since_detection > SKIP_INITIAL_FRAMES:
                # Check if we have enough history
                if len(point_history) == HISTORY_LENGTH:
                    # Preprocess the point history
                    pre_processed_history = pre_process_point_history(frame, point_history)
                    
                    # Write to CSV
                    csv_writer.writerow([class_id] + pre_processed_history)
                    collected_count += 1
                    
                    # Draw visualization
                    debug_image = draw_point_history(debug_image, list(point_history))
                    
                    # Get finger name for display
                    finger_names = {4: "Thumb", 8: "Index", 12: "Middle", 16: "Ring", 20: "Pinky"}
                    finger_name = finger_names.get(finger_landmark, f"Landmark {finger_landmark}")
                    
                    # Add status text
                    status_text = f"{class_name} | Tracking: {finger_name} | Frame: {frame_count}/{total_frames} | Collected: {collected_count}/{MAX_FRAMES_PER_VIDEO}"
                    cv.putText(debug_image, status_text, (10, 30),
                             cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv.LINE_AA)
                    cv.putText(debug_image, status_text, (10, 30),
                             cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
        else:
            # No hand detected
            point_history.append([0, 0])
            
            if hand_detected:
                # If hand was previously detected but now lost, show waiting message
                cv.putText(debug_image, "Waiting for hand...", (10, 30),
                         cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
        
        # Display the frame
        cv.namedWindow("Point History Data Collection", cv.WINDOW_NORMAL)
        cv.resizeWindow("Point History Data Collection", 480, 640)
        cv.imshow("Point History Data Collection", debug_image)
        
        # Check for key press (ESC to stop, 'q' to quit)
        key = cv.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or 'q'
            print("      Stopped by user")
            cap.release()
            return collected_count
    
    cap.release()
    print(f"      ✓ Collected {collected_count} samples from this video")
    return collected_count


# =====================================================
# MAIN PROCESSING
# =====================================================

def main():
    print("=" * 70)
    print("POINT HISTORY DATA COLLECTION FROM VIDEOS")
    print("=" * 70)
    print(f"Target hand: {TARGET_HAND}")
    print(f"Max frames per video: {MAX_FRAMES_PER_VIDEO}")
    print(f"Frame interval: {FRAME_INTERVAL}")
    print(f"Skip initial frames: {SKIP_INITIAL_FRAMES}")
    print(f"History length: {HISTORY_LENGTH}")
    print(f"Output CSV: {OUTPUT_CSV}")
    print("-" * 70)
    
    # Get all video files organized by class
    video_files = get_video_files(VIDEO_ROOT)
    
    if not video_files:
        print("No video files found! Please check the VIDEO_ROOT path.")
        return
    
    print(f"Found {len(video_files)} classes:")
    for class_name, videos in video_files.items():
        print(f"  - {class_name}: {len(videos)} videos")
    print("-" * 70)
    
    # Open CSV file for writing
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        total_samples = 0
        total_videos = sum(len(videos) for videos in video_files.values())
        processed_videos = 0
        
        # Process each class
        for class_id, (class_name, videos) in enumerate(video_files.items()):
            print(f"\n[Class {class_id}] Processing '{class_name}' ({len(videos)} videos)")
            class_samples = 0
            
            for video_index, video_path in enumerate(videos):
                samples = process_video(video_path, class_id, class_name, 
                                      video_index, len(videos), writer)
                class_samples += samples
                total_samples += samples
                processed_videos += 1
                
                # Check if window was closed
                if cv.getWindowProperty("Point History Data Collection", cv.WND_PROP_VISIBLE) < 1:
                    print("\nWindow closed by user")
                    break
            
            print(f"  Class '{class_name}' completed: {class_samples} total samples")
            
            # Check if window was closed
            if cv.getWindowProperty("Point History Data Collection", cv.WND_PROP_VISIBLE) < 1:
                break
    
    cv.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total videos processed: {processed_videos}/{total_videos}")
    print(f"Total samples collected: {total_samples}")
    print(f"CSV saved to: {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()