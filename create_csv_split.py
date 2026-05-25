import os
import pandas as pd
import random

BASE_DIR = "./MonolayerFFF/Images"
SPLIT_RATIO = 0.8

class_map = {
    "D1": 0,
    "D2": 1,
    "H": 2
}

data = []

for cls in class_map:
    folder = os.path.join(BASE_DIR, cls)
    for file in os.listdir(folder):
        if file.endswith(".png"):
            data.append({
                "filepath": os.path.join(folder, file),
                "label": class_map[cls]
            })

# Shuffle
random.shuffle(data)

# Split
split_idx = int(len(data) * SPLIT_RATIO)
train_data = data[:split_idx]
val_data = data[split_idx:]

# Save CSV
pd.DataFrame(train_data).to_csv("./data/train.csv", index=False)
pd.DataFrame(val_data).to_csv("./data/val.csv", index=False)

print("CSV split created.")
