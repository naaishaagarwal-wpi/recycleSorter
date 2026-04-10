import os
import shutil

# Paths
original_dataset = "/Users/naaisha/.cache/kagglehub/datasets/mostafaabla/garbage-classification/versions/1/garbage_classification"
labelled_dataset = "/Users/naaisha/junior_year/mams/Com Sci/independentCSProject/recyclingSorter/dataset/labelled_dataset"

# Define all category folders
categories = [
    "paper", "cardboard", "metal", "green-glass", "brown-glass", "white-glass",
    "biological", "clothes", "shoes", "battery", "trash"
]

# Create each category folder inside the destination directory
for folder in categories:
    os.makedirs(os.path.join(labelled_dataset, folder), exist_ok=True)

# Function to copy images for each category
def copy_images(categories):
    for folder in categories:
        src_folder = os.path.join(original_dataset, folder)
        dst_folder = os.path.join(labelled_dataset, folder)
        
        if not os.path.exists(src_folder):
            print(f"Skipping missing folder: {src_folder}")
            continue
        
        for filename in os.listdir(src_folder):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                shutil.copy(
                    os.path.join(src_folder, filename),
                    os.path.join(dst_folder, filename)
                )

# Copy images for all categories
copy_images(categories)

print("✅ Done! Dataset is organized by category in:", labelled_dataset)

