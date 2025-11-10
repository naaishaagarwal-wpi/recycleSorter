import os
import shutil

# Paths
original_dataset = "/Users/naaisha/.cache/kagglehub/datasets/mostafaabla/garbage-classification/versions/1/garbage_classification"

binary_dataset = "/Users/naaisha/junior_year/mams/Com Sci/independentCSProject/recyclingSorter/dataset/binary_dataset"

# Define which folders go into which category
recyclable_folders = ["paper", "cardboard", "metal", "green-glass", "brown-glass", "white-glass"]  # adjust as needed
not_recylable_folders = ["biological", "clothes", "shoes", "battery", "trash"]        # adjust as needed

# Create binary folders
os.makedirs(os.path.join(binary_dataset, "Recyclable"), exist_ok=True)
os.makedirs(os.path.join(binary_dataset, "Not_Recylable"), exist_ok=True)

# Function to copy images
def copy_images(src_folders, dst_folder):
    for folder in src_folders:
        folder_path = os.path.join(original_dataset, folder)
        if not os.path.exists(folder_path):
            continue
        for filename in os.listdir(folder_path):
            if filename.endswith((".jpg", ".png", ".jpeg")):
                shutil.copy(
                    os.path.join(folder_path, filename),
                    os.path.join(dst_folder, filename)
                )

copy_images(recyclable_folders, os.path.join(binary_dataset, "Recyclable"))
copy_images(not_recylable_folders, os.path.join(binary_dataset, "Not_Recylable"))

print("Done! Binary dataset is ready.")
