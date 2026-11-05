import json
import os

base_dir = "/lustre/scratch/WUR/ABGC/kland001/benchmark/data/cow_kpt_320"
images_dir = os.path.join(base_dir, "images")
ann_dir = os.path.join(base_dir, "annotations")
os.makedirs(images_dir, exist_ok=True)
os.makedirs(ann_dir, exist_ok=True)

train_root = "/home/WUR/kland001/retraining/data/cow_barn_kpt_it2_corr_300_20/train"
val_root = "/home/WUR/kland001/retraining/data/cow_barn_kpt_it2_corr_300_20/val"

with open(f"{train_root}/annotations/instances_default.json") as f:
    train_data = json.load(f)
with open(f"{val_root}/annotations/instances_default.json") as f:
    val_data = json.load(f)

# Symlink all train images
for f in os.listdir(f"{train_root}/images"):
    src = os.path.abspath(f"{train_root}/images/{f}")
    dst = os.path.join(images_dir, f)
    if not os.path.exists(dst):
        os.symlink(src, dst)

# Symlink all val images
for f in os.listdir(f"{val_root}/images"):
    src = os.path.abspath(f"{val_root}/images/{f}")
    dst = os.path.join(images_dir, f)
    if not os.path.exists(dst):
        os.symlink(src, dst)

# Build combined COCO JSON
train_img_ids = {img["id"] for img in train_data["images"]}
train_ann_ids = {ann["id"] for ann in train_data["annotations"]}

combined = {
    "info": train_data.get("info", {}),
    "licenses": train_data.get("licenses", []),
    "categories": train_data["categories"],
    "images": list(train_data["images"]),
    "annotations": list(train_data["annotations"])
}

img_id_offset = max(train_img_ids) + 1 if train_img_ids else 1000000
ann_id_offset = max(train_ann_ids) + 1 if train_ann_ids else 1000000

val_id_map = {}
for img in val_data["images"]:
    orig_id = img["id"]
    if orig_id in train_img_ids:
        new_id = img_id_offset
        img_id_offset += 1
    else:
        new_id = orig_id
    val_id_map[orig_id] = new_id
    img_copy = dict(img)
    img_copy["id"] = new_id
    combined["images"].append(img_copy)

for ann in val_data["annotations"]:
    ann_copy = dict(ann)
    if ann["id"] in train_ann_ids:
        ann_copy["id"] = ann_id_offset
        ann_id_offset += 1
    ann_copy["image_id"] = val_id_map[ann["image_id"]]
    combined["annotations"].append(ann_copy)

out_ann_path = os.path.join(ann_dir, "instances_all_320.json")
with open(out_ann_path, "w") as f:
    json.dump(combined, f)

print(f"Successfully created combined COCO dataset at {out_ann_path}")
print(f"Total images: {len(combined['images'])}")
print(f"Total annotations: {len(combined['annotations'])}")
print(f"Total images in directory: {len(os.listdir(images_dir))}")
