# train_with_total_npy.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# =====================
# Dataset 정의
# =====================
class GestureSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.y[idx]

# =====================
# CNN-LSTM 모델 정의
# =====================
class CNNLSTM(nn.Module):
    def __init__(self, input_size=99, hidden_size=64, num_classes=12):  # 0~11
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
        out = out[:, -1, :]  # 마지막 타임스텝
        return self.fc(out)

# =====================
# 학습 함수
# =====================
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
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
    print("[INFO] 데이터 불러오는 중...")
    X = np.load('./gesture_data/X_total.npy')  # shape (N, 30, 99)
    y = np.load('./gesture_data/y_total.npy')  # shape (N,)

    dataset = GestureSequenceDataset(X, y)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = CNNLSTM(num_classes=len(np.unique(y)))  # 자동 클래스 수 계산 (12개일 가능성 높음)
    train_model(model, train_loader, val_loader, epochs=20)

    torch.save(model.state_dict(), './gesture_model_moving_test.pt')
    print("[✅ 저장 완료] 모델이 gesture_model_final.pt 로 저장되었습니다.")
