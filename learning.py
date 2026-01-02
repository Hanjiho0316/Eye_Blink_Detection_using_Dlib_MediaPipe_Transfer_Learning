import os
import cv2
import dlib
import torch
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Albumentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------
# 1️⃣ 환경설정
# -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREDICTOR_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\shape_predictor_68_face_landmarks.dat"
cew_root = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\CEW"

SAVE_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\best_multitask_model.pth"
LOG_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\train_multitask_log.csv"

EPOCHS = 30
BATCH_SIZE = 8
EYE_CROP_PADDING = 20
REGRESSION_WEIGHT = 0.5  # MSE 손실 가중치 λ

# -------------------
# 2️⃣ Dlib + MediaPipe 초기화
# -------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def extract_features(image):
    if image is None:
        return np.zeros(68*2), np.zeros(468*3)
        
    h, w, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray)
    
    # Dlib
    dlib_pts = np.zeros(68*2)
    if len(rects) > 0:
        shape = predictor(gray, rects[0])
        dlib_coords = []
        for p in shape.parts():
            dlib_coords.extend([p.x / w, p.y / h])
        dlib_pts = np.array(dlib_coords)

    # MediaPipe
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)
    
    mp_pts = np.zeros(468*3)
    if results.multi_face_landmarks:
        mp_coords = []
        for lm in results.multi_face_landmarks[0].landmark:
            mp_coords.extend([lm.x, lm.y, lm.z])
        mp_pts = np.array(mp_coords)

    return dlib_pts, mp_pts

# -------------------
# 3️⃣ Dataset Class
# -------------------
class EyeDataset(Dataset):
    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.detector = detector
        self.predictor = predictor

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        if img is None:
            dummy_img = torch.zeros((3, 224, 224), dtype=torch.float32)
            dummy_feat = torch.zeros((68*2 + 468*3), dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            return dummy_img, dummy_feat, dummy_label

        dlib_f, mp_f = extract_features(img)
        features = np.concatenate([dlib_f, mp_f])

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray)
        eye_crop_img = img
        if len(rects) > 0:
            shape = self.predictor(gray, rects[0])
            eye_coords_x = [shape.part(i).x for i in range(36,48)]
            eye_coords_y = [shape.part(i).y for i in range(36,48)]
            x_min = max(0, min(eye_coords_x)-EYE_CROP_PADDING)
            x_max = min(img.shape[1], max(eye_coords_x)+EYE_CROP_PADDING)
            y_min = max(0, min(eye_coords_y)-EYE_CROP_PADDING)
            y_max = min(img.shape[0], max(eye_coords_y)+EYE_CROP_PADDING)
            if x_max > x_min and y_max > y_min:
                eye_crop_img = img[y_min:y_max, x_min:x_max]

        eye_crop_rgb = cv2.cvtColor(eye_crop_img, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=eye_crop_rgb)
            eye_crop_tensor = augmented['image']
        else:
            eye_crop_tensor = torch.from_numpy(eye_crop_rgb).permute(2,0,1).float()/255.0

        return eye_crop_tensor, torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# -------------------
# 4️⃣ Albumentations Transform
# -------------------
train_transform = A.Compose([
    A.Resize(224,224),
    A.OneOf([
        A.RandomBrightness(limit=0.3, p=1.0),
        A.RandomContrast(limit=0.3, p=1.0),
        A.RandomGamma(gamma_limit=(80,120), p=1.0)
    ], p=0.5),
    A.HorizontalFlip(p=0.5),
    A.GaussNoise(var_limit=(10.0,50.0), p=0.3),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# -------------------
# 5️⃣ Multimodal Model
# -------------------
class MultiModalBlinkModel(nn.Module):
    def __init__(self, img_dim=512, feature_dim=68*2 + 468*3):
        super().__init__()
        base_model = models.resnet18(pretrained=True)
        base_model.fc = nn.Identity()
        self.cnn = base_model

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim,512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.blink_fc = nn.Sequential(
            nn.Linear(512+img_dim,256),
            nn.ReLU(),
            nn.Linear(256,2) # 눈 감김 상태
        )

        # 눈 위치 회귀용 (feature만)
        self.regression_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 68*2)  # Dlib 눈 좌표 예측
        )

    def forward(self, img, features):
        img_feat = self.cnn(img)
        feat = self.feature_fc(features)
        combined = torch.cat((img_feat, feat), dim=1)
        blink_out = self.blink_fc(combined)
        reg_out = self.regression_fc(features)
        return blink_out, reg_out

# -------------------
# 6️⃣ 메인 학습 루프
# -------------------
if __name__ == "__main__":
    # 데이터 로드
    open_imgs = [os.path.join(cew_root,"OpenEye",f) for f in os.listdir(os.path.join(cew_root,"OpenEye")) if f.lower().endswith(('.jpg','.png'))]
    closed_imgs = [os.path.join(cew_root,"ClosedEye",f) for f in os.listdir(os.path.join(cew_root,"ClosedEye")) if f.lower().endswith(('.jpg','.png'))]

    img_paths = open_imgs + closed_imgs
    labels = [1]*len(open_imgs) + [0]*len(closed_imgs)

    train_paths, val_paths, train_labels, val_labels = train_test_split(img_paths, labels, test_size=0.2, stratify=labels, random_state=42)

    train_dataset = EyeDataset(train_paths, train_labels, train_transform)
    val_dataset = EyeDataset(val_paths, val_labels, val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiModalBlinkModel().to(DEVICE)

    # 손실함수
    criterion_blink = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    best_val_loss = float('inf')
    log_records = []

    for epoch in range(1,EPOCHS+1):
        model.train()
        train_loss = 0
        for imgs, feats, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            imgs, feats, labels_batch = imgs.to(DEVICE), feats.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            blink_out, reg_out = model(imgs, feats)

            # 눈 감김 + 눈 좌표 회귀 손실 합산
            loss_blink = criterion_blink(blink_out, labels_batch)
            # 실제 눈 좌표 (Dlib 좌표) 사용
            true_coords = feats[:,:68*2]  # Dlib eye 좌표 부분만 회귀 타겟
            loss_reg = criterion_reg(reg_out, true_coords)
            loss = loss_blink + REGRESSION_WEIGHT*loss_reg

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct,total=0,0
        with torch.no_grad():
            for imgs, feats, labels_batch in val_loader:
                imgs, feats, labels_batch = imgs.to(DEVICE), feats.to(DEVICE), labels_batch.to(DEVICE)
                blink_out, reg_out = model(imgs, feats)
                loss_blink = criterion_blink(blink_out, labels_batch)
                true_coords = feats[:,:68*2]
                loss_reg = criterion_reg(reg_out, true_coords)
                loss = loss_blink + REGRESSION_WEIGHT*loss_reg
                val_loss += loss.item()
                pred = torch.argmax(blink_out, dim=1)
                correct += (pred==labels_batch).sum().item()
                total += labels_batch.size(0)
        val_loss /= len(val_loader)
        val_acc = correct/total

        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print("  -> Best model saved.")

        log_records.append({"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"val_acc":val_acc})

    pd.DataFrame(log_records).to_csv(LOG_PATH,index=False)
    print("Training Finished.")
