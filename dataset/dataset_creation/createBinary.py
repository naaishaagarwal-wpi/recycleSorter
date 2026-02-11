import os
import shutil

# Paths
original_dataset = "dataset/garbage_labelled"

binary_dataset = "/Users/naaisha/junior_year/mams/Com Sci/independentCSProject/recycleSorter/dataset/binary_dataset"

# Define which folders go into which category
recyclable_folders = ["cardboard", "glass", "metal", "paper", "plastic"]  # adjust as needed
not_recylable_folders = ["battery", "biological", "clothes", "shoes", "trash"]        # adjust as needed

# Create binary folders
os.makedirs(os.path.join(binary_dataset, "Recyclable"), exist_ok=True)
os.makedirs(os.path.join(binary_dataset, "Not_Recyclable"), exist_ok=True)

# Function to copy images
def copy_images(src_folders, dst_folder):
    for folder in src_folders:
        folder_path = os.path.join(original_dataset, folder)
        if not os.path.exists(folder_path):
            continue

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            src = os.path.join(folder_path, filename)
            dst = os.path.join(dst_folder, filename)

            # If file exists, add suffix
            if os.path.exists(dst):
                name, ext = os.path.splitext(filename)
                i = 1
                while os.path.exists(os.path.join(dst_folder, f"{name}_{i}{ext}")):
                    i += 1
                dst = os.path.join(dst_folder, f"{name}_{i}{ext}")

            shutil.copy(src, dst)


copy_images(not_recylable_folders, os.path.join(binary_dataset, "Not_Recyclable"))

print("Done! Binary dataset is ready.")
