import os
import cv2
import xml.etree.ElementTree as ET
import albumentations as A

# Paths
ANNOTATION_FILE = "cvat.xml"               # your CVAT export
IMAGE_DIR = "preprocessed_dataset"         # your dataset folder (matches CVAT names)
OUTPUT_DIR = "augmented_dataset"           # save augmented images + labels here
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parse CVAT XML
tree = ET.parse(ANNOTATION_FILE)
root = tree.getroot()

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5)
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

# Loop through images in annotation
for image_tag in root.findall('image'):
    img_name = image_tag.attrib['name']
    width = int(image_tag.attrib['width'])
    height = int(image_tag.attrib['height'])

    # Path to image inside preprocessed_dataset
    img_path = os.path.join(IMAGE_DIR, img_name)

    if not os.path.exists(img_path):
        print(f"⚠ Image {img_name} not found in {IMAGE_DIR}")
        continue

    # Read image
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠ Failed to load {img_path}")
        continue

    # Collect bounding boxes and labels
    bboxes = []
    labels = []
    for box in image_tag.findall('box'):
        xtl = float(box.attrib['xtl'])
        ytl = float(box.attrib['ytl'])
        xbr = float(box.attrib['xbr'])
        ybr = float(box.attrib['ybr'])
        label = box.attrib['label']
        bboxes.append([xtl, ytl, xbr, ybr])
        labels.append(label)

    if not bboxes:
        continue

    # Apply augmentation
    transformed = transform(image=image, bboxes=bboxes, category_ids=labels)

    aug_img = transformed['image']
    aug_bboxes = transformed['bboxes']
    aug_labels = transformed['category_ids']

    # Save augmented image
    base_name = os.path.splitext(img_name)[0]
    out_img_name = f"{base_name}_aug.jpg"
    out_img_path = os.path.join(OUTPUT_DIR, out_img_name)
    cv2.imwrite(out_img_path, aug_img)

    # Save augmented annotation in YOLO format
    out_label_name = f"{base_name}_aug.txt"
    out_label_path = os.path.join(OUTPUT_DIR, out_label_name)
    with open(out_label_path, "w") as f:
        for bbox, label in zip(aug_bboxes, aug_labels):
            x_min, y_min, x_max, y_max = bbox
            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height
            f.write(f"{label} {x_center} {y_center} {w} {h}\n")

    print(f"✅ Augmented {img_name} -> {out_img_name}")

print("🎉 Data augmentation completed! Augmented files saved in", OUTPUT_DIR)

