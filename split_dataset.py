import os
import shutil
import random

# Paths
dataset_dir = "COVID-19_Radiography_Dataset"
classes = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]

for cls in classes:
    # ✅ Look inside "images" folder of each class
    class_path = os.path.join(dataset_dir, cls, "images")

    if not os.path.exists(class_path):
        print(f"⚠️ Skipping {cls}, no 'images' folder found.")
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    random.shuffle(images)

    split_idx = int(0.8 * len(images))
    train_files = images[:split_idx]
    test_files = images[split_idx:]

    # Train & test dirs
    train_class_dir = os.path.join("train", cls)
    test_class_dir = os.path.join("test", cls)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    for img in train_files:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))
    for img in test_files:
        shutil.copy(os.path.join(class_path, img), os.path.join(test_class_dir, img))

    print(f"✅ {cls}: {len(train_files)} train, {len(test_files)} test images copied.")

print("🎉 Dataset successfully split into train/ and test/ folders.")
