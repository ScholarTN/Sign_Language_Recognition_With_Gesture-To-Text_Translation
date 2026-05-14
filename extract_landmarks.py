import os
import cv2
import numpy as np
import pandas as pd
DATASET_PATH = "/home/scholar/Documents/Projects/ASL_Sign_Language_Recognition/Sign_Language_Recognition_With_Gesture-To-Text_Translation/asl_data/asl_alphabet_train/asl_alphabet_train"

rows = []

IMG_SIZE = 64

print("Folders found:")
print(os.listdir(DATASET_PATH)[:5])

for label in os.listdir(DATASET_PATH):

    folder = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder):
        continue

    print(f"\nProcessing label: {label}")

    files = os.listdir(folder)

    print(f"Found {len(files)} files")

    count = 0

    for file in files:

        path = os.path.join(folder, file)

        image = cv2.imread(path)

        if image is None:
            print(f"FAILED TO READ: {path}")
            continue

        try:

            # Resize
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            # Grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Blur
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # Flatten
            features = blur.flatten()

            row = list(features)

            row.append(label)

            rows.append(row)

            count += 1

        except Exception as e:

            print(f"ERROR PROCESSING {path}")
            print(e)

        # Limit per class for speed
        if count >= 120:
            break

    print(f"Successfully processed: {count}")

print(f"\nTOTAL ROWS: {len(rows)}")

# Save only if data exists
if len(rows) > 0:

    columns = [f"p{i}" for i in range(IMG_SIZE * IMG_SIZE)]

    columns.append("label")

    df = pd.DataFrame(rows, columns=columns)

    df.to_csv("asl_landmarks.csv", index=False)

    print("CSV saved successfully.")

else:

    print("NO DATA EXTRACTED.")