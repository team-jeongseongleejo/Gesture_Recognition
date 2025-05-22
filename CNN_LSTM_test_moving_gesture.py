import cv2
import mediapipe as mp
import numpy as np
import torch
from gesture_code.train_with_total_npy import CNNLSTM  # 학습 시 사용한 정의를 import

import torch.nn.functional as F
import requests
import time
import threading

# ========== 사용자 설정 ==========
model_path = './gesture_model_moving_test.pt'
seq_length = 10

label_map_inv = {
    0: 'one', 1: 'two', 2: 'thumbs_up', 3: 'thumbs_down', 4: 'ok',
    5: 'small_heart', 6: 'three', 7: 'three', 8: 'thumbs_left',
    9: 'thumbs_right', 10: 'four', 11: 'clockwise'
}

label_map = {v: k for k, v in label_map_inv.items()}
clockwise_label = label_map['clockwise']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 네트워크 설정 ==========
server_url = 'http://192.168.160.204:5000/gesture'

def send_to_server(gesture_label):
    data = {"gesture": gesture_label}
    try:
        response = requests.post(server_url, json=data)
        if response.status_code == 200:
            print(f"[서버 응답] 상태 코드: {response.status_code} (OK)")
            print(f"[서버 메시지] {response.text}")
        else:
            print(f"[경고] 서버가 200이 아닌 상태 코드 반환: {response.status_code}")
    except Exception as e:
        print(f"[에러] 서버 전송 실패: {e}")

# ========== 모델 로딩 ==========
model = CNNLSTM(num_classes=len(label_map_inv))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ========== MediaPipe 설정 ==========
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# ========== 실시간 캡처 ==========
cap = cv2.VideoCapture(0)
seq = []
action_seq = []
prev_joint = None

print("[INFO] 웹캠 실행 중. 'q'를 눌러 종료하세요.")

last_sent_gesture = None
last_sent_time = 0
gesture_delay = 2  # 새로운 제스처 등장 시 최소 유지 시간 (초)
send_interval = 3  # 같은 제스처일 때 재전송 주기 (초)
confirm_frame_count = 3  # 최소 유지되어야 하는 프레임 수

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))

            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            if prev_joint is not None:
                movement = np.linalg.norm(joint[:, :3] - prev_joint[:, :3])
            else:
                movement = 0
            prev_joint = joint.copy()

            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
            v = v2 - v1
            v /= np.linalg.norm(v, axis=1)[:, np.newaxis]
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])
            seq.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            if len(seq) < seq_length:
                continue

            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            input_tensor = torch.tensor(input_data).to(device)

            with torch.no_grad():
                y_pred = model(input_tensor)
                pred_class = torch.argmax(y_pred, dim=1).item()

            if pred_class == clockwise_label and movement < 0.015:
                continue

            gesture = label_map_inv[pred_class]
            cv2.putText(img, gesture.upper(), (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
            print(f'[예측] {gesture.upper()}')

            action_seq.append(gesture)
            if len(action_seq) > confirm_frame_count:
                action_seq.pop(0)

            # 최근 프레임 중 모두 같은 제스처인 경우만 전송 고려
            if action_seq.count(gesture) == len(action_seq):
                current_time = time.time()

                if gesture != last_sent_gesture:
                    if current_time - last_sent_time >= gesture_delay:
                        threading.Thread(target=send_to_server, args=(gesture,), daemon=True).start()
                        last_sent_gesture = gesture
                        last_sent_time = current_time

                elif current_time - last_sent_time >= send_interval:
                    threading.Thread(target=send_to_server, args=(gesture,), daemon=True).start()
                    last_sent_time = current_time

    else:
        # 🔻 손이 없을 때 시퀀스 초기화
        seq.clear()
        action_seq.clear()
        prev_joint = None

    cv2.imshow('Gesture Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
