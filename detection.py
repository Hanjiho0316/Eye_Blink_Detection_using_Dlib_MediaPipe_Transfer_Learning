import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn as nn
from torchvision import models
import dlib
import mediapipe as mp
import math
# 사운드 재생을 위한 라이브러리 추가
from playsound import playsound 

# -------------------
# 1️⃣ 설정
# -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 경로를 본인 환경에 맞게 수정하세요
PREDICTOR_PATH = "/Users/hanjiho/Desktop/eye detect/eye_blink_detector-master/face recognition/shape_predictor_68_face_landmarks.dat"
SAVE_PATH = "/Users/hanjiho/Desktop/eye detect/eye_blink_detector-master/face recognition/best_multitask_model.pth"

# ⚠️ 사운드 파일 경로를 사용자의 실제 경고음 파일 경로로 수정하세요!
WARNING_SOUND_PATH = "/Users/hanjiho/Desktop/warning_sound.wav" 

# 경고 각도 임계값
ANGLE_THRESHOLD = 30 
# 측면 얼굴 인식을 위해 여백을 넉넉히 줌
EYE_CROP_PADDING = 40 

# -------------------
# 2️⃣ dlib + MediaPipe 초기화
# -------------------
try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except RuntimeError as e:
    print(f"dlib predictor 로드 오류: {e}")
    exit()

mp_face = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True 
)

# -------------------
# 3️⃣ Multimodal Model
# -------------------
class MultiModalBlinkModel(nn.Module):
    def __init__(self, img_dim=512, feature_dim=68*2 + 468*3): 
        super().__init__()
        base_model = models.resnet18(pretrained=False) 
        base_model.fc = nn.Identity()
        self.cnn = base_model 

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 512), 
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.blink_fc = nn.Sequential(
            nn.Linear(512 + img_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, img, features):
        img_feat = self.cnn(img)
        feat = self.feature_fc(features)
        combined = torch.cat((img_feat, feat), dim=1)
        blink_out = self.blink_fc(combined)
        return blink_out

# -------------------
# 4️⃣ 모델 로드
# -------------------
feature_dim = 68*2 + 468*3 
model = MultiModalBlinkModel(feature_dim=feature_dim).to(DEVICE)
try:
    model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE))
    model.eval() 
    print(f"모델 로드 완료: {SAVE_PATH}")
except FileNotFoundError:
    print(f"오류: 모델 파일이 없습니다. 경로를 확인하세요: {SAVE_PATH}")
    exit()

# -------------------
# 5️⃣ Transform 정의
# -------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# -------------------
# 6️⃣ 웹캠 실행
# -------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

blink_flag = False
blink_count = 0
# ⚠️ 사운드 중복 재생 방지를 위한 플래그 추가
is_warning_active = False 

print("웹캠 실행 중... (ESC 키를 누르면 종료됩니다)")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1) 
    h, w, _ = frame.shape 
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = mp_face.process(rgb_frame)
    
    face_detected = False
    
    # 초기화
    head_pitch = 0.0
    head_yaw = 0.0
    head_roll = 0.0
    gaze_text = "N/A" 
    
    img_tensor = torch.zeros((1, 3, 224, 224), dtype=torch.float32).to(DEVICE)
    features_tensor = torch.zeros((1, feature_dim), dtype=torch.float32).to(DEVICE)
    
    if results.multi_face_landmarks:
        face_detected = True
        landmarks = results.multi_face_landmarks[0].landmark
        
        # --- 0. dlib shape 생성 ---
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for lm in landmarks: 
            x, y = int(lm.x * w), int(lm.y * h)
            x_min = min(x_min, x); y_min = min(y_min, y)
            x_max = max(x_max, x); y_max = max(y_max, y)
        
        mp_rect = dlib.rectangle(max(0, x_min - 5), max(0, y_min - 5), min(w - 1, x_max + 5), min(h - 1, y_max + 5))
        
        try:
            shape = predictor(gray_frame, mp_rect)
        except Exception:
            face_detected = False
            continue 

        # --- 1. 이미지 크롭 (dlib 'shape') ---
        eye_coords_x = []
        eye_coords_y = []
        for i in range(36, 48):
            eye_coords_x.append(shape.part(i).x)
            eye_coords_y.append(shape.part(i).y)
        
        # 넓은 패딩 적용
        x_min_crop = max(0, min(eye_coords_x) - EYE_CROP_PADDING)
        x_max_crop = min(frame.shape[1], max(eye_coords_x) + EYE_CROP_PADDING)
        y_min_crop = max(0, min(eye_coords_y) - EYE_CROP_PADDING)
        y_max_crop = min(frame.shape[0], max(eye_coords_y) + EYE_CROP_PADDING)
        
        if x_max_crop > x_min_crop and y_max_crop > y_min_crop:
            eye_crop_img = frame[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
            cv2.rectangle(frame, (x_min_crop, y_min_crop), (x_max_crop, y_max_crop), (255, 255, 0), 2)
            
            rgb_crop = cv2.cvtColor(eye_crop_img, cv2.COLOR_BGR2RGB)
            if rgb_crop.size != 0:
                img_tensor = transform(rgb_crop).unsqueeze(0).to(DEVICE)
            else:
                face_detected = False
        else:
            face_detected = False 

        # --- 2. 특징 준비 & Geometry Pose Estimation ---
        if face_detected:
            # (1) dlib Feature
            dlib_coords = []
            for p in shape.parts():
                dlib_coords.extend([p.x / w, p.y / h]) 
            dlib_f = np.array(dlib_coords)

            # (2) MediaPipe Feature
            mp_coords = []
            for i in range(468):
                lm = landmarks[i]
                mp_coords.extend([lm.x, lm.y, lm.z]) 
            mp_f = np.array(mp_coords)

            # -------------------------------------------------------
            # 🆕 기하학적 Head Pose Estimation (PnP 대체)
            # -------------------------------------------------------
            p33  = np.array([landmarks[33].x * w,  landmarks[33].y * h])  # 왼쪽 눈 끝
            p263 = np.array([landmarks[263].x * w, landmarks[263].y * h]) # 오른쪽 눈 끝
            p1   = np.array([landmarks[1].x * w,   landmarks[1].y * h])   # 코 끝
            p61  = np.array([landmarks[61].x * w,  landmarks[61].y * h])  # 입 왼쪽
            p291 = np.array([landmarks[291].x * w, landmarks[291].y * h]) # 입 오른쪽

            # [1] Roll (기울기)
            dy = p263[1] - p33[1]
            dx = p263[0] - p33[0]
            head_roll = np.degrees(np.arctan2(dy, dx))
            
            # [2] Yaw (좌우)
            eye_center = (p33 + p263) / 2
            face_width = np.linalg.norm(p263 - p33)
            
            if face_width > 0:
                yaw_scale = 150 
                head_yaw = ((p1[0] - eye_center[0]) / face_width) * yaw_scale

            # [3] Pitch (상하)
            mouth_center = (p61 + p291) / 2
            face_height = np.linalg.norm(eye_center - mouth_center)
            
            if face_height > 0:
                mid_y = (eye_center[1] + mouth_center[1]) / 2
                pitch_scale = 150
                head_pitch = ((p1[1] - mid_y) / face_height) * pitch_scale

            # [시각화] 코 끝에서 바라보는 방향 선 그리기
            nose_pt = (int(p1[0]), int(p1[1]))
            cv2.line(frame, nose_pt, (int(p1[0] + head_yaw * 2), int(p1[1])), (0, 255, 255), 3)
            cv2.line(frame, nose_pt, (int(p1[0]), int(p1[1] + head_pitch * 2)), (255, 0, 255), 3)
            # -------------------------------------------------------

            # --- 3. 시선 추정 (Pupil) ---
            try:
                outer = landmarks[33].x
                inner = landmarks[133].x
                pupil = landmarks[473].x
                eye_w = abs(inner - outer)
                if eye_w > 0: 
                    gaze_val = ((pupil - outer) / eye_w - 0.5) * 2.0
                    gaze_text = f"{gaze_val:.2f}"
            except Exception:
                gaze_text = "Err"
            
            # 특징 벡터 결합
            features = np.concatenate([dlib_f, mp_f]) 
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
    else:
        face_detected = False

    # --- 4. 모델 추론 (깜빡임) ---
    pred = 1 
    if face_detected: 
        with torch.no_grad():
            blink_out = model(img_tensor, features_tensor)
            pred = torch.argmax(blink_out, dim=1).item() 

    # --- 5. 깜빡임 카운트 ---
    if pred == 0: 
        if not blink_flag:
            blink_flag = True
    else: 
        if blink_flag:
            blink_flag = False
            blink_count += 1

    # -------------------------------------------------------
    ## 🚨 --- 7. 경고 시스템 및 사운드 (추가된 부분) ---
    # -------------------------------------------------------
    warning_message = ""
    warning_color = (0, 0, 0) # 기본 (검은색)
    
    is_unstable = abs(head_pitch) > ANGLE_THRESHOLD or abs(head_roll) > ANGLE_THRESHOLD
    
    if is_unstable:
        warning_message = f"warning: Danger angle exceeded (Pitch/Roll > {ANGLE_THRESHOLD}°)"
        warning_color = (0, 0, 255) # 빨간색 경고
        
        # 🚨 사운드 재생: is_warning_active가 False일 때만 재생 (반복 방지)
        if not is_warning_active:
            is_warning_active = True
            try:
                # playsound는 비동기(non-blocking)를 지원하지 않으므로, 
                # 짧은 소리 파일만 사용하거나 새로운 쓰레드로 실행하는 것이 좋으나, 
                # 가장 간단한 형태로 구현합니다.
                playsound(WARNING_SOUND_PATH, block=False)
                print("경고음 재생됨!")
            except Exception as e:
                print(f"경고음 재생 오류: {e} (파일 경로를 확인하세요)")
    else:
        # 안정 범위로 돌아오면 플래그 초기화
        is_warning_active = False 

    # 화면 우측 상단에 경고 메시지 표시
    cv2.putText(frame, warning_message, (w - 500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2)
    # -------------------------------------------------------


    # --- 6. 화면 표시 ---
    if not face_detected:
        status_text = "No Face"
        status_color = (0, 0, 255)
    else:
        status_text = "Closed" if pred == 0 else "Open"
        status_color = (0, 0, 255) if pred == 0 else (0, 255, 0)
    
    # 텍스트 표시 UI
    cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # 얼굴이 감지되면 항상 포즈 정보 표시
    info_color = (0, 255, 255) if face_detected else (128, 128, 128)
    
    cv2.putText(frame, f"Yaw(L/R): {head_yaw:.1f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
    cv2.putText(frame, f"Pitch(U/D): {head_pitch:.1f}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
    cv2.putText(frame, f"Roll(Tilt): {head_roll:.1f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)
    cv2.putText(frame, f"Gaze: {gaze_text}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_color, 2)

    cv2.imshow("Eye Blink & Geometry Pose", frame)
    if cv2.waitKey(1) & 0xFF == 27: 
        break

cap.release()
mp_face.close()
cv2.destroyAllWindows()
print(f"Final Blink Count: {blink_count}")
