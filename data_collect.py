import os
import csv
import cv2 as cv
import numpy as np
import mediapipe as mp

# Optional: suppress verbose TensorFlow and MediaPipe logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# 1. CONFIGURE PATHS AND CLASS ORDER
DATA_ROOT   = "C:\\Users\\mendez\\Desktop\\Mendez\\archive\\Collated"  
OUTPUT_CSV  = "model/fsl_classifier/keypoint.csv"
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
CLASSES     = sorted(
    d for d in os.listdir(DATA_ROOT)
    if os.path.isdir(os.path.join(DATA_ROOT, d))
)

# 2. INITIALIZE MediaPipe Hands
mp_hands   = mp.solutions.hands
hands      = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# UTILITIES

def calc_bounding_rect(image, landmarks):
    img_h, img_w = image.shape[:2]
    pts = []
    for lm in landmarks.landmark:
        x = min(int(lm.x * img_w), img_w - 1)
        y = min(int(lm.y * img_h), img_h - 1)
        pts.append([x, y])
    pts = np.array(pts, dtype=np.int32)
    x, y, w, h = cv.boundingRect(pts)
    return [x, y, x + w, y + h]


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness,
                   hand_sign_text, finger_gesture_text, alphabet_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info = ''
    if handedness:
        info = handedness.classification[0].label
    if hand_sign_text:
        info += ':' + hand_sign_text
    if alphabet_text:
        info += ' (' + alphabet_text + ')'
    cv.putText(image, info, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def draw_landmarks(image, landmark_point):
    if len(landmark_point) == 0:
        return image
    # Draw lines
    connections = [
        (2,3), (3,4), (5,6), (6,7), (7,8),
        (9,10), (10,11), (11,12), (13,14), (14,15), (15,16),
        (17,18), (18,19), (19,20),
        (0,1), (1,2), (2,5), (5,9), (9,13), (13,17), (17,0)
    ]
    for (start, end) in connections:
        cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[start]), tuple(landmark_point[end]), (255, 255, 255), 2)
    # Draw keypoints
    for i, lm in enumerate(landmark_point):
        radius = 8 if i in {4,8,12,16,20} else 5
        cv.circle(image, tuple(lm), radius, (255, 255, 255), -1)
        cv.circle(image, tuple(lm), radius, (0, 0, 0), 1)
    return image

# 3. OPEN CSV FOR WRITING
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    for class_id, cls in enumerate(CLASSES):
        cls_dir = os.path.join(DATA_ROOT, cls)
        images = sorted(os.listdir(cls_dir))
        total  = len(images)
        for idx, fname in enumerate(images):
            img_path = os.path.join(cls_dir, fname)
            img = cv.imread(img_path)
            if img is None:
                continue
            print(f"[{class_id+1}/{len(CLASSES)}] '{cls}' image {idx+1}/{total}")
            img_rgb  = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            result   = hands.process(img_rgb)
            annotated = img.copy()
            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0]
                pts = [(min(int(pt.x * img.shape[1]), img.shape[1]-1),
                        min(int(pt.y * img.shape[0]), img.shape[0]-1))
                       for pt in lm.landmark]
                brect = calc_bounding_rect(annotated, lm)
                handed = result.multi_handedness[0] if result.multi_handedness else None
                annotated = draw_bounding_rect(True, annotated, brect)
                annotated = draw_landmarks(annotated, pts)
                annotated = draw_info_text(annotated, brect, handed, "", "", f"{cls} {idx+1}/{total}")
                # Extract & preprocess
                coords = np.array(pts, dtype=np.float32)
                base   = coords[0]
                rel    = coords - base
                flat   = rel.flatten()
                mv     = np.max(np.abs(flat))
                if mv > 0:
                    flat = flat / mv
                writer.writerow([class_id] + flat.tolist())
            cv.imshow("Data Collection", annotated)
            if cv.waitKey(10) & 0xFF == ord('q'):
                break
        if cv.getWindowProperty("Data Collection", cv.WND_PROP_VISIBLE) < 1:
            break

cv.destroyAllWindows()
print("Processing complete. CSV saved to", OUTPUT_CSV)
