import cv2
import os
from tqdm import tqdm

# Paths for input folders
custom_folder = r"C:\Users\kumku\Downloads\VIT Downloads\cv_project\manual_dataset"
online_folder = r"C:\Users\kumku\Downloads\VIT Downloads\cv_project\online_dataset"

# Path for output folder
output_folder = r"C:\Users\kumku\Downloads\VIT Downloads\cv_project\preprocessed_dataset"
os.makedirs(output_folder, exist_ok=True)

# Target size for resizing (change if needed)
target_size = (224, 224)

# Collect all images from both folders
all_images = []
for folder in [custom_folder, online_folder]:
    if os.path.exists(folder):
        for file in os.listdir(folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(folder, file))

# Counters
processed_count = 0
skipped_count = 0

# Process each image
for idx, img_path in enumerate(tqdm(all_images, desc="Preprocessing Images")):
    try:
        # Read image
        img = cv2.imread(img_path)
        
        if img is None:  
            skipped_count += 1
            continue  # Skip if image is not readable

        # Resize
        img = cv2.resize(img, target_size)

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Save preprocessed image
        save_path = os.path.join(output_folder, f"image_{idx}.jpg")
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        processed_count += 1

    except Exception as e:
        skipped_count += 1
        print(f"❌ Skipped {img_path}: {e}")

# Final Summary
print("\n✅ Preprocessing Completed!")
print(f"Total images found: {len(all_images)}")
print(f"Processed successfully: {processed_count}")
print(f"Skipped (corrupted/unreadable): {skipped_count}")
print(f"Saved preprocessed images in: {output_folder}")