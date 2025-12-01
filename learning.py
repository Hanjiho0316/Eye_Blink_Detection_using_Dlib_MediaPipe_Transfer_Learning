import os
import cv2
import dlib
import torch
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from torchvision import models
# transforms는 이제 albumentations를 사용하므로 torchvision.transforms는 일부 기능만 남기거나 제거해도 됩니다.
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# -------------------
# 🆕 Albumentations 추가
# -------------------
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------
# 1️⃣ 설정
# -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREDICTOR_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\shape_predictor_68_face_landmarks.dat"
cew_root = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\CEW"

SAVE_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\best_multitask_model.pth"
LOG_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognitiontrain_multitask_log.csv"

EPOCHS = 30
EYE_CROP_PADDING = 20 

# -------------------
# 2️⃣ dlib + MediaPipe Feature Extractor
# -------------------
detector = dlib.get_frontal_face_detector()
try:
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
except RuntimeError:
    print(f"Error: dlib predictor '{PREDICTOR_PATH}'를 찾을 수 없습니다.")
    exit()
    
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def extract_features(image):
    if image is None:
        return np.zeros(68*2), np.zeros(468*3)
        
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    
    dlib_pts = np.zeros(68 * 2) 
    if len(rects) > 0:
        shape = predictor(gray, rects[0])
        dlib_coords = []
        for p in shape.parts():
            dlib_coords.extend([p.x / w, p.y / h]) 
        dlib_pts = np.array(dlib_coords)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)
    
    mp_pts = np.zeros(468 * 3) 
    if results.multi_face_landmarks:
        mp_coords = []
        for lm in results.multi_face_landmarks[0].landmark:
            mp_coords.extend([lm.x, lm.y, lm.z])
        mp_pts = np.array(mp_coords)

    return dlib_pts, mp_pts

# -------------------
# 3️⃣ Dataset Class (🔴 Albumentations 적용 수정)
# -------------------
class EyeDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(PREDICTOR_PATH)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        # 1. 이미지 로드
        img = cv2.imread(img_path) 
        if img is None:
            # 예외 처리
            dummy_img = torch.zeros((3, 224, 224), dtype=torch.float32)
            dummy_feat = torch.zeros((68*2 + 468*3), dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_img, dummy_feat, dummy_label

        # 2. 랜드마크 추출
        dlib_f, mp_f = extract_features(img)
        features = np.concatenate([dlib_f, mp_f])

        # 3. 눈 영역 크롭
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray)
        
        eye_crop_img = img # 기본값
        
        if len(rects) > 0:
            shape = self.predictor(gray, rects[0])
            eye_coords_x = []
            eye_coords_y = []
            for i in range(36, 48): 
                eye_coords_x.append(shape.part(i).x)
                eye_coords_y.append(shape.part(i).y)
            
            x_min = max(0, min(eye_coords_x) - EYE_CROP_PADDING)
            x_max = min(img.shape[1], max(eye_coords_x) + EYE_CROP_PADDING)
            y_min = max(0, min(eye_coords_y) - EYE_CROP_PADDING)
            y_max = min(img.shape[0], max(eye_coords_y) + EYE_CROP_PADDING)
            
            if x_max > x_min and y_max > y_min:
                eye_crop_img = img[y_min:y_max, x_min:x_max]

        # 4. Transform 적용 (Albumentations)
        # BGR -> RGB 변환 (Albumentations 및 학습을 위해)
        eye_crop_rgb = cv2.cvtColor(eye_crop_img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            # Albumentations는 키워드 인자 image=... 를 사용하고 결과를 dict로 반환합니다.
            augmented = self.transform(image=eye_crop_rgb)
            eye_crop_tensor = augmented['image']
        else:
            # Transform이 없는 경우 기본 텐서 변환
            eye_crop_tensor = torch.from_numpy(eye_crop_rgb).permute(2, 0, 1).float() / 255.0

        return eye_crop_tensor, torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# -------------------
# 4️⃣ Multimodal Model
# -------------------
class MultiModalBlinkModel(nn.Module):
    def __init__(self, img_dim=512, feature_dim=68*2 + 468*3):
        super().__init__()
        base_model = models.resnet18(pretrained=True)
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

# ----------------------------------------------------
# 💥💥💥 메인 실행 블록 💥💥💥
# ----------------------------------------------------
if __name__ == '__main__':

    # -------------------
    # 데이터 로드
    # -------------------
    open_imgs = [os.path.join(cew_root, "OpenEye", f) for f in os.listdir(os.path.join(cew_root, "OpenEye")) if f.lower().endswith(('.jpg','.png'))]
    closed_imgs = [os.path.join(cew_root, "ClosedEye", f) for f in os.listdir(os.path.join(cew_root, "ClosedEye")) if f.lower().endswith(('.jpg','.png'))]

    img_paths = open_imgs + closed_imgs
    labels = [1]*len(open_imgs) + [0]*len(closed_imgs) 

    print(f"Total: {len(img_paths)} (Open: {len(open_imgs)}, Closed: {len(closed_imgs)})")

    # 가중치 계산
    count_0 = len(closed_imgs)
    count_1 = len(open_imgs)
    class_weights = None
    if count_0 > 0 and count_1 > 0:
        total = count_0 + count_1
        weight_0 = total / (2.0 * count_0)
        weight_1 = total / (2.0 * count_1)
        class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float32).to(DEVICE)
        print(f"Weights applied - 0: {weight_0:.2f}, 1: {weight_1:.2f}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # -------------------
    # 🆕 Tunnel Augmentation 적용 (핵심)
    # -------------------
    train_transform = A.Compose([
        A.Resize(224, 224),
        
        # [터널/저조도 시뮬레이션]
        # p=0.5: 50% 확률로 어둡게 만듭니다.
        A.OneOf([
            # Gamma Correction: 비선형적으로 어둡게 만들어 실제 어둠과 비슷하게 함
            A.RandomGamma(gamma_limit=(120, 240), p=1.0), 
            # Brightness/Contrast: 밝기와 대비를 동시에 낮춤
            A.RandomBrightnessContrast(brightness_limit=(-0.6, -0.2), contrast_limit=(-0.6, -0.2), p=1.0),
        ], p=0.5),

        # [센서 노이즈 및 흔들림 시뮬레이션]
        # p=0.3: 30% 확률로 노이즈나 블러 추가
        A.OneOf([
            # ISO Noise: 감도 증가로 인한 컬러 노이즈
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            # Gauss Noise: 일반적인 거친 노이즈
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            # Motion Blur: 운전 중 진동/저조도 셔터 스피드 저하
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # [기본 증강]
        A.HorizontalFlip(p=0.5),
        
        # [정규화 및 텐서 변환]
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # Dataset 생성
    train_dataset = EyeDataset(train_paths, train_labels, train_transform)
    val_dataset = EyeDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0) # Windows 오류 시 num_workers=0
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    # -------------------
    # 학습 루프
    # -------------------
    model = MultiModalBlinkModel(feature_dim=68*2 + 468*3).to(DEVICE)
    criterion_blink = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    log_records = []

    print(f"Training started on {DEVICE}...")

    for epoch in range(1, EPOCHS+1):
        model.train()
        total_blink_loss = 0
        
        for eye_crop_img, feat, label in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            eye_crop_img, feat, label = eye_crop_img.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
            
            optimizer.zero_grad()
            blink_out = model(eye_crop_img, feat)
            
            loss = criterion_blink(blink_out, label)
            loss.backward()
            optimizer.step()
            
            total_blink_loss += loss.item()
            
        total_blink_loss /= len(train_loader)

        # Validation
        model.eval()
        val_blink_loss = 0
        correct, total = 0,0
        with torch.no_grad():
            for eye_crop_img, feat, label in val_loader:
                eye_crop_img, feat, label = eye_crop_img.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
                
                blink_out = model(eye_crop_img, feat)
                
                loss = criterion_blink(blink_out, label)
                val_blink_loss += loss.item()
                
                pred = torch.argmax(blink_out, dim=1)
                correct += (pred==label).sum().item()
                total += label.size(0)
                
        val_blink_loss /= len(val_loader)
        val_acc = correct / total
        
        print(f"Epoch {epoch}/{EPOCHS} - Train Blink: {total_blink_loss:.4f} | Val Blink: {val_blink_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_blink_loss < best_val_loss:
            best_val_loss = val_blink_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Best model saved.")

        log_records.append({
            "epoch": epoch,
            "train_blink_loss": total_blink_loss,
            "val_blink_loss": val_blink_loss,
            "val_acc": val_acc
        })

    df_log = pd.DataFrame(log_records)
    df_log.to_csv(LOG_PATH, index=False)
    print("Training finished.")
