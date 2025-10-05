import os
import csv
import copy
import itertools
import cv2 as cv
import numpy as np
import mediapipe as mp

# Optional: suppress verbose TensorFlow and MediaPipe logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# 1. CONFIGURE PATHS AND CLASS ORDER
DATA_ROOT   = "C:\\Users\\Dell\\Downloads\\archive\\Collated"  

# CONFIGURATION
TARGET_HAND = "left"  # Set to "Right" or "Left" - hands will be flipped to match this
MAX_ROTATION_ATTEMPTS = 12  # Will try rotations from -30 to +30 degrees in 5-degree steps
ROTATION_STEP = 5  # Degrees per rotation attempt

OUTPUT_CSV  = "model/fsl_classifier/{TARGET_HAND}/keypoint.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
CLASSES     = sorted(
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
)

# 2. INITIALIZE MediaPipe Hands
mp_hands   = mp.solutions.hands
hands      = mp_hands.Hands(
    static_image_mode=True,  # Changed to True for images
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# UTILITIES - COPIED EXACTLY FROM app.py

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image


def draw_info_text(image, brect, handedness, hand_sign_text,
                   finger_gesture_text, alphabet_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    info_text = ""
    if handedness:
        info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "":
        info_text = info_text + ':' + hand_sign_text
    if alphabet_text != "":
        info_text = info_text + ' (' + alphabet_text + ')'
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


# NEW UTILITY FUNCTIONS

def rotate_image(image, angle):
    """Rotate image by given angle in degrees"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(image, M, (w, h), flags=cv.INTER_LINEAR, 
                           borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated


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


def process_image_with_rotation_retry(image, hands_model, class_name, image_index, total_images):
    """Process image with rotation retry until hand is detected"""
    original_image = image.copy()
    
    for attempt in range(MAX_ROTATION_ATTEMPTS + 1):  # +1 for original image (0 rotation)
        if attempt == 0:
            # Try original image first
            current_image = original_image
            rotation_angle = 0
        else:
            # Try rotations from -30 to +30 degrees
            rotation_angle = (attempt - 1) * ROTATION_STEP - (MAX_ROTATION_ATTEMPTS // 2) * ROTATION_STEP
            current_image = rotate_image(original_image, rotation_angle)
        
        # Process with MediaPipe
        img_rgb = cv.cvtColor(current_image, cv.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
        result = hands_model.process(img_rgb)
        img_rgb.flags.writeable = True
        
        if result.multi_hand_landmarks is not None:
            hand_landmarks = result.multi_hand_landmarks[0]
            handedness = result.multi_handedness[0] if result.multi_handedness else None
            
            # Get detected hand label
            detected_hand_label = handedness.classification[0].label if handedness else "Unknown"
            
            # Calculate landmarks and bounding rect
            brect = calc_bounding_rect(current_image, hand_landmarks)
            landmark_list = calc_landmark_list(current_image, hand_landmarks)
            
            # Check if we need to flip the hand
            if should_flip_hand(detected_hand_label, TARGET_HAND):
                # Flip the image horizontally
                current_image = cv.flip(current_image, 1)
                # Flip the landmarks
                landmark_list = flip_landmarks_horizontally(landmark_list, current_image.shape[1])
                # Update the handedness label for display
                detected_hand_label = TARGET_HAND
                # Recalculate bounding rect for flipped landmarks
                # Create a dummy landmarks object for bounding rect calculation
                class DummyLandmark:
                    def __init__(self, x, y):
                        self.x = x
                        self.y = y
                
                class DummyLandmarks:
                    def __init__(self, landmark_list, img_width, img_height):
                        self.landmark = []
                        for point in landmark_list:
                            self.landmark.append(DummyLandmark(point[0] / img_width, point[1] / img_height))
                
                dummy_landmarks = DummyLandmarks(landmark_list, current_image.shape[1], current_image.shape[0])
                brect = calc_bounding_rect(current_image, dummy_landmarks)
            
            # Preprocess landmarks
            pre_processed_landmark_list = pre_process_landmark(landmark_list)
            
            # Create debug image for visualization
            debug_image = copy.deepcopy(current_image)
            debug_image = draw_bounding_rect(True, debug_image, brect)
            debug_image = draw_landmarks(debug_image, landmark_list)
            
            # Create dummy handedness for display
            class DummyHandedness:
                def __init__(self, label):
                    self.classification = [type('obj', (object,), {'label': label})()]
            
            display_handedness = DummyHandedness(detected_hand_label)
            
            status_text = f"{class_name} {image_index+1}/{total_images}"
            if rotation_angle != 0:
                status_text += f" (rotated {rotation_angle}°)"
            if detected_hand_label != handedness.classification[0].label if handedness else "Unknown":
                status_text += " (flipped)"
            
            debug_image = draw_info_text(
                debug_image,
                brect,
                display_handedness,
                "",  # hand_sign_text
                "",  # finger_gesture_text  
                status_text  # alphabet_text
            )
            
            print(f"  ✓ Hand detected after {attempt} attempts (rotation: {rotation_angle}°)")
            if rotation_angle != 0:
                print(f"    Applied rotation: {rotation_angle}°")
            if should_flip_hand(handedness.classification[0].label if handedness else "Unknown", TARGET_HAND):
                print(f"    Flipped {handedness.classification[0].label if handedness else 'Unknown'} hand to {TARGET_HAND}")
            
            return pre_processed_landmark_list, debug_image, True
        
        print(f"  Attempt {attempt + 1}: No hand detected (rotation: {rotation_angle}°)")
    
    print(f"  ✗ Failed to detect hand after {MAX_ROTATION_ATTEMPTS + 1} attempts")
    return None, original_image, False


# 3. MAIN PROCESSING LOOP
print(f"Starting data collection for {TARGET_HAND} hand training")
print(f"Will attempt up to {MAX_ROTATION_ATTEMPTS + 1} rotations per image to detect hands")
print(f"Rotation range: -{MAX_ROTATION_ATTEMPTS // 2 * ROTATION_STEP}° to +{MAX_ROTATION_ATTEMPTS // 2 * ROTATION_STEP}° in {ROTATION_STEP}° steps")
print("-" * 60)

with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    
    total_processed = 0
    total_successful = 0
    total_failed = 0

    for class_id, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATA_ROOT, cls)
        images = sorted(os.listdir(cls_dir))
        total_images = len(images)
        
        print(f"\n[{class_id+1}/{len(CLASSES)}] Processing class '{cls}' ({total_images} images)")
        
        class_successful = 0
        
        for idx, fname in enumerate(images):
            img_path = os.path.join(cls_dir, fname)
            img = cv.imread(img_path)
            if img is None:
                print(f"  ✗ Could not load image: {fname}")
                total_failed += 1
                continue
                
            total_processed += 1
            
            # Process image with rotation retry
            processed_landmarks, debug_image, success = process_image_with_rotation_retry(
                img, hands, cls, idx, total_images
            )
            
            if success:
                # Write to CSV
                writer.writerow([class_id] + processed_landmarks)
                total_successful += 1
                class_successful += 1
            else:
                total_failed += 1
            
            # Display the debug image
            cv.imshow("Data Collection", debug_image)
            if cv.waitKey(30) & 0xFF == ord('q'):
                print("\nStopped by user")
                break
                
        print(f"  Class '{cls}' completed: {class_successful}/{total_images} successful")
        
        # Check if window was closed
        if cv.getWindowProperty("Data Collection", cv.WND_PROP_VISIBLE) < 1:
            break

cv.destroyAllWindows()

print("\n" + "=" * 60)
print("PROCESSING COMPLETE")
print("=" * 60)
print(f"Total images processed: {total_processed}")
print(f"Successfully processed: {total_successful}")
print(f"Failed to process: {total_failed}")
print(f"Success rate: {(total_successful/total_processed*100):.1f}%")
print(f"CSV saved to: {OUTPUT_CSV}")
print(f"Target hand configuration: {TARGET_HAND}")
print("=" * 60)