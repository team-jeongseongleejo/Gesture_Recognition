# train_gesture.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# =====================
# Dataset 정의
# =====================
class GestureDataset(Dataset):
    def __init__(self, np_data, seq_length=30):
        self.seq_length = seq_length
        self.data = []
        self.labels = []
        for i in range(len(np_data) - seq_length):
            seq = np_data[i:i+seq_length, :-1]  # features
            label = int(np_data[i+seq_length-1, -1])  # 마지막 프레임 기준 레이블
            self.data.append(seq)
            self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# =====================
# CNN-LSTM 모델 정의
# =====================
class CNNLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=64, num_classes=17):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, 30, 99) -> (B, 99, 30)
        x = self.relu(self.conv1(x))  # (B, 64, 30)
        x = x.transpose(1, 2)  # (B, 30, 64)
        out, _ = self.lstm(x)  # (B, 30, H)
        out = out[:, -1, :]  # 마지막 타임스텝만
        return self.fc(out)

# =====================
# 학습 함수
# =====================
def train_model(model, train_loader, val_loader, epochs=15, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[{epoch+1}/{epochs}] Loss: {total_loss / len(train_loader):.4f}")

        # 검증
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                preds = model(x_val).argmax(dim=1)
                correct += (preds == y_val).sum().item()
                total += y_val.size(0)
        print(f"Validation Accuracy: {correct / total:.4f}")

# =====================
# 실행 코드
# =====================
if __name__ == '__main__':
    data = np.load('../gesture_data/combined_static_plus_dynamic_17-2.npy', allow_pickle=True)

    # =====================
    # ✅ 정규화 적용
    # =====================
    features = data[:, :-1]  # 99개 입력 특성
    labels = data[:, -1].reshape(-1, 1)  # 레이블은 그대로
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    data = np.hstack([features_scaled, labels])  # 다시 합쳐서 전체 배열 구성

    # =====================
    # Dataset 및 학습
    # =====================
    dataset = GestureDataset(data)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = CNNLSTM()
    train_model(model, train_loader, val_loader, epochs=15)
    torch.save(model.state_dict(), './gesture_model17-2.pt')
