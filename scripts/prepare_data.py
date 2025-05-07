import yaml, os, shutil, glob, logging
from datetime import datetime

os.makedirs("logs", exist_ok=True)


log_file = os.path.join("logs", f"data_preparation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logging.info("Starting data preparation...")

yaml_path = "/media/dhruv/Local Disk/Yolo-Lite/Yolo-Lite/FurnitureDetection2.v3i.yolov8/data.yaml"
try:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
except Exception as e:
    logging.error(f"Failed to load YAML file: {e}")
    exit(1)

classes = cfg['names']
os.makedirs("data/processed", exist_ok=True)

class_file_path = "data/processed/classes.names"
with open(class_file_path, "w") as f:
    f.write("\n".join(classes))
logging.info(f"✅ Saved class names to {class_file_path}")

for subset in ["train", "val"]:
    subset_dir = cfg[subset]
    logging.info(f"Processing {subset} set from {subset_dir}")

    img_files = glob.glob(os.path.join(subset_dir, "*.jpg")) + glob.glob(os.path.join(subset_dir, "*.png"))

    out_txt_path = f"data/processed/{subset}.txt"
    with open(out_txt_path, "w") as out:
        for img_path in img_files:
            out.write(img_path + "\n")

            lbl = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"

            try:
                shutil.copy(img_path, "data/processed/")
                shutil.copy(lbl, "data/processed/")
            except FileNotFoundError:
                logging.warning(f" Missing label file for: {img_path}")
            except Exception as e:
                logging.error(f" Failed to copy {img_path} or its label: {e}")
    
    logging.info(f"Finished writing {subset}.txt and copying files.")

logging.info("Data preparation complete.")
