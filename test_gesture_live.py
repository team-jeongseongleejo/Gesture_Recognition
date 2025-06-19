import cv2
import mediapipe as mp
import numpy as np
import torch
import time
import threading
import requests
from CNN_LSTM_train_live_gesture import CNNLSTM

# ==================== 사용자 설정 ====================
model_path = './gesture_model_final-2.pt'
seq_length = 30
server_url = 'http://192.168.253.204:5000/gesture'

label_map_inv = {
    0: 'one', 1: 'two', 2: 'thumbs_up', 3: 'thumbs_down', 4: 'ok',
    5: 'small_heart', 6: 'three', 7: 'three', 8: 'thumbs_left',
    9: 'thumbs_right', 10: 'four', 11: 'promise', 12: 'vertical_V',
    13: 'horizontal_V', 14: 'spider_man', 15: 'rotate', 16: 'rotate', 17: 'nothing', 18: 'side_moving'
}
label_map = {v: k for k, v in label_map_inv.items()}
sequence_labels = ['rotate', 'rotate', 'side_moving']

clockwise_label = label_map['rotate']
side_moving_label = label_map['side_moving']

gesture_delay = 2         # 새로운 제스처 전송 최소 유지 시간
send_interval = 3         # 같은 제스처 반복 전송 간격
confirm_frame_count = 3   # 최소 유지 프레임 수
last_sent_gesture = None
last_sent_time = 0
send_feedback = ""        # 화면 출력 메시지
send_feedback_time = 0

# ==================== 서버 전송 함수 ====================
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

# ==================== 모델 로딩 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNLSTM(num_classes=len(label_map_inv))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ==================== MediaPipe ====================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ==================== 실시간 캡처 ====================
cap = cv2.VideoCapture(0)
seq = []
action_seq = []
prev_joint = None
gesture_hold_start_time = None

print("[INFO] 웹캠 실행 중. 'q'를 눌러 종료하세요.")

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

            movement = 0
            if prev_joint is not None:
                movement = np.linalg.norm(joint[:, :3] - prev_joint[:, :3])
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
                confidence = torch.nn.functional.softmax(y_pred, dim=1)[0][pred_class].item()
                action_seq.append(pred_class)

            if len(action_seq) >= confirm_frame_count and all(x == action_seq[-1] for x in action_seq[-confirm_frame_count:]):
                gesture = label_map_inv[pred_class]

                if gesture in sequence_labels:
                    if movement < 0.02:
                        gesture_hold_start_time = None
                        continue
                    if gesture_hold_start_time is None:
                        gesture_hold_start_time = time.time()
                        continue
                    if confidence < 0.5:
                        continue
                    elif time.time() - gesture_hold_start_time < 1.5:
                        continue
                else:
                    gesture_hold_start_time = None
                    if confidence < 0.85:
                        continue

                # ========= 출력 및 서버 전송 =========
                cv2.putText(img, f'{gesture.upper()} ({confidence:.2f})', (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                print(f'[예측] {gesture.upper()} (Conf: {confidence:.2f})')

                current_time = time.time()

                if gesture != last_sent_gesture:
                    if current_time - last_sent_time >= gesture_delay:
                        threading.Thread(target=send_to_server, args=(gesture,), daemon=True).start()
                        last_sent_gesture = gesture
                        last_sent_time = current_time
                        send_feedback = f"sent: {gesture.upper()}"
                        send_feedback_time = current_time
                elif current_time - last_sent_time >= send_interval:
                    threading.Thread(target=send_to_server, args=(gesture,), daemon=True).start()
                    last_sent_time = current_time
                    send_feedback = f"re-sent: {gesture.upper()}"
                    send_feedback_time = current_time

    else:
        seq.clear()
        action_seq.clear()
        prev_joint = None
        gesture_hold_start_time = None

    # ===== 화면에 전송 피드백 표시 (5초 유지) =====
    if time.time() - send_feedback_time < 5:
        cv2.putText(img, send_feedback, (20, 450), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow('Gesture Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
