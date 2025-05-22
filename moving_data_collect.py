import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# ✨ 사용자 설정
gesture_label = 'clockwise'
save_path = f'./gesture_data/main_data/data_{gesture_label}.csv'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

seq_length = 30
max_num_hands = 1

# MediaPipe 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=max_num_hands,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# 버퍼 및 저장용 리스트
seq_buffer = []
collected_sequences = []

# 웹캠 연결
cap = cv2.VideoCapture(0)
print(f'[INFO] Start collecting "{gesture_label}" gesture data. Press "q" to stop.')

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        continue

    #img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for res in result.multi_hand_landmarks:
            joint = np.zeros((21, 4))

            for j, lm in enumerate(res.landmark):
                joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

            # 벡터 및 각도 계산
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3]
            v = v2 - v1
            v /= np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:],
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
            angle = np.degrees(angle)

            d = np.concatenate([joint.flatten(), angle])
            seq_buffer.append(d)

            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # 시퀀스 완성되면 한 줄로 저장
            if len(seq_buffer) == seq_length:
                seq_np = np.array(seq_buffer).flatten()  # shape: (30×99,)
                seq_labeled = np.append(seq_np, gesture_label)
                collected_sequences.append(seq_labeled)
                seq_buffer = []  # 초기화

    cv2.imshow('Collecting Clockwise Gesture', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ✨ 저장
collected_sequences = np.array(collected_sequences)
df = pd.DataFrame(collected_sequences)
df.to_csv(save_path, index=False)
print(f'[INFO] 시퀀스 저장 완료: {save_path} / shape = {collected_sequences.shape}')
