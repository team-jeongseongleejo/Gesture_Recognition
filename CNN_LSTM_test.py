# test_gesture_live.py
import cv2
import mediapipe as mp
import numpy as np
import torch
from CNN_LSTM_train import CNNLSTM  # 모델 정의는 재사용

# ========== 사용자 설정 ==========
model_path = './gesture_model11.pt'
seq_length = 30

label_map_inv = {0: 'one', 1: 'two', 2: 'thumbs_up', 3: 'thumbs_down', 4: 'ok', 5: 'small_heart', 6:'three',
                 7: 'three', 8: 'thumbs_left', 9:'thumbs_right', 10:'four'}

# ========== 모델 로딩 ==========
model = CNNLSTM()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
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

            # 벡터 기반 관절 간 각도 계산
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

            # 시퀀스 길이가 부족하면 continue
            if len(seq) < seq_length:
                continue

            # 예측 수행
            input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
            input_tensor = torch.tensor(input_data)
            with torch.no_grad():
                y_pred = model(input_tensor)
                pred_class = torch.argmax(y_pred, dim=1).item()
                action_seq.append(pred_class)

            # 연속된 예측 결과가 같으면 출력
            if len(action_seq) >= 3 and all(x == action_seq[-1] for x in action_seq[-3:]):
                gesture = label_map_inv[pred_class]
                cv2.putText(img, gesture.upper(), (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
                print(f'[예측] {gesture.upper()}')

    cv2.imshow('Gesture Recognition', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
