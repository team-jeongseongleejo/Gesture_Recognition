# combine_gesture_data_fixed.py

import os
import pandas as pd
import numpy as np

# 경로 설정
csv_dir = './gesture_data/main_data'
output_path = '.gesture_data/combined_gesture_data8-2.npy'

label_map = {
    'one': 0,
    'two': 1,
    'thumbs_up': 2,
    'thumbs_down': 3,
    'okay' : 4,
    'thumbs_left': 5,
    'thumbs_right': 6,
    'clockwise': 7
    }

csv_files = {
    'one': 'data_one.csv',
    'two': 'data_two.csv',
    'thumbs_up': 'data_thumbs_up.csv',
    'thumbs_down': 'data_thumbs_down.csv',
    'okay': 'data_ok.csv',
    'thumbs_left': 'data_thumbs_left.csv',
    'thumbs_right': 'data_thumbs_right.csv',
    'clockwise': 'data_clockwise_converted.csv'
}

all_data = []

for label_str, file_name in csv_files.items():
    file_path = os.path.join(csv_dir, file_name)
    if not os.path.exists(file_path):
        print(f"[경고] 파일 없음: {file_path}")
        continue

    df = pd.read_csv(file_path, header=None)
    df[df.columns[-1]] = label_map[label_str]  # 마지막 열을 숫자 레이블로

    all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
combined_df = combined_df.sample(frac=1).reset_index(drop=True)

# float32로 명확하게 변환!
data_np = combined_df.to_numpy().astype(np.float32)
np.save(output_path, data_np)

print(f"[✅ 저장 완료] {output_path} (shape={data_np.shape})")
