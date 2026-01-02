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

# ------------------- 설정 -------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PREDICTOR_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\shape_predictor_68_face_landmarks.dat"
cew_root = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\CEW"

SAVE_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\best_multitask_model.pth"
LOG_PATH = r"C:\Users\FORYOUCOM\Desktop\CT preprocessing\face recognition\train_multitask_log.csv"

EPOCHS = 30
EYE_CROP_PADDING = 20 

# ------------------- dlib + MediaPipe 초기화 -------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
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

# ------------------- Dataset -------------------
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
        img = cv2.imread(img_path)
        if img is None:
            dummy_img = torch.zeros((3,224,224), dtype=torch.float32)
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
            x_min = max(0, min(eye_coords_x) - EYE_CROP_PADDING)
            x_max = min(img.shape[1], max(eye_coords_x) + EYE_CROP_PADDING)
            y_min = max(0, min(eye_coords_y) - EYE_CROP_PADDING)
            y_max = min(img.shape[0], max(eye_coords_y) + EYE_CROP_PADDING)
            if x_max > x_min and y_max > y_min:
                eye_crop_img = img[y_min:y_max, x_min:x_max]

        eye_crop_rgb = cv2.cvtColor(eye_crop_img, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=eye_crop_rgb)
            eye_crop_tensor = augmented['image']
        else:
            eye_crop_tensor = torch.from_numpy(eye_crop_rgb).permute(2,0,1).float()/255.0

        return eye_crop_tensor, torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ------------------- 데이터 증강 -------------------
train_transform = A.Compose([
    A.Resize(224,224),

    # 밝기/대비 / 감마 증강
    A.OneOf([
        A.RandomGamma(gamma_limit=(120,240), p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
    ], p=0.5),

    # 노이즈/블러
    A.OneOf([
        A.ISONoise(color_shift=(0.01,0.05), intensity=(0.1,0.5), p=1.0),
        A.GaussNoise(var_limit=(10.0,50.0), p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.3),

    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2()
])

# ------------------- Multimodal 모델 -------------------
class MultiModalBlinkModel(nn.Module):
    def __init__(self, img_dim=512, feature_dim=68*2+468*3):
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
            nn.Linear(512+img_dim, 256),
            nn.ReLU(),
            nn.Linear(256,2)
        )

    def forward(self, img, features):
        img_feat = self.cnn(img)
        feat = self.feature_fc(features)
        combined = torch.cat((img_feat, feat), dim=1)
        blink_out = self.blink_fc(combined)
        return blink_out

# ------------------- 학습/검증 -------------------
if __name__=='__main__':
    open_imgs = [os.path.join(cew_root,"OpenEye",f) for f in os.listdir(os.path.join(cew_root,"OpenEye")) if f.lower().endswith(('.jpg','.png'))]
    closed_imgs = [os.path.join(cew_root,"ClosedEye",f) for f in os.listdir(os.path.join(cew_root,"ClosedEye")) if f.lower().endswith(('.jpg','.png'))]

    img_paths = open_imgs + closed_imgs
    labels = [1]*len(open_imgs) + [0]*len(closed_imgs)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = EyeDataset(train_paths, train_labels, train_transform)
    val_dataset = EyeDataset(val_paths, val_labels, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

    model = MultiModalBlinkModel(feature_dim=68*2+468*3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1,31):
        model.train()
        for img, feat, label in train_loader:
            img, feat, label = img.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
            optimizer.zero_grad()
            output = model(img, feat)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct,total=0,0
        with torch.no_grad():
            for img, feat, label in val_loader:
                img, feat, label = img.to(DEVICE), feat.to(DEVICE), label.to(DEVICE)
                output = model(img, feat)
                pred = torch.argmax(output, dim=1)
                correct += (pred==label).sum().item()
                total += label.size(0)
        val_acc = correct/total
        print(f"Epoch {epoch} - Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), SAVE_PATH)
