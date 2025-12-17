import json
import random
from pathlib import Path
from collections import defaultdict
import argparse

from ultralytics import YOLO
import shutil
import os
from datetime import datetime

ROOT = Path(__file__).resolve().parent
TRAIN_ROOT = ROOT / "yolo_train_data"
IMAGES_DIR = TRAIN_ROOT / "images"
META_FILE = TRAIN_ROOT / "annotations.jsonl"

DATASET_ROOT = ROOT / "yolo_dataset"
IMG_TRAIN = DATASET_ROOT / "images" / "train"
IMG_VAL = DATASET_ROOT / "images" / "val"
LBL_TRAIN = DATASET_ROOT / "labels" / "train"
LBL_VAL = DATASET_ROOT / "labels" / "val"

RANDOM_SEED = 42
VAL_FRACTION = 0.2  # 20% av bilderna blir validering


def build_dataset():
    """
    Läser annotations.jsonl och bygger ett YOLO-dataset med train/val-split.

    - Ignorerar label "__IGNORE__"
    - Gruppar per bild, så alla boxar för samma bild hamnar i samma split
    - Skriver data.yaml med rätt train/val paths
    """
    if not META_FILE.exists():
        print("Hittar inte", META_FILE)
        return None, None

    # skapa mappar
    for p in [IMG_TRAIN, IMG_VAL, LBL_TRAIN, LBL_VAL]:
        p.mkdir(parents=True, exist_ok=True)
        for f in p.glob("*"):
            f.unlink()

    # 1) Läs alla records och gruppera per bild
    image_records = defaultdict(list)  # img_name -> [rec, rec, ...]
    with META_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            img_name = rec["image"]
            image_records[img_name].append(rec)

    if not image_records:
        print("Inga records hittades i", META_FILE)
        return None, None

    # 2) Klassnamn → id
    class_to_id = {}
    next_id = 0

    # 3) Slumpa train/val-split på bildnivå
    img_names = list(image_records.keys())
    random.Random(RANDOM_SEED).shuffle(img_names)

    n_val = max(1, int(len(img_names) * VAL_FRACTION)) if len(img_names) > 5 else 1
    val_set = set(img_names[:n_val])
    train_set = set(img_names[n_val:]) if len(img_names) > 1 else set()

    if not train_set:
        # Om väldigt få bilder, lägg alla i train men varna
        print("Väldigt få bilder – använder alla som train.")
        train_set = set(img_names)
        val_set = set()

    # För statistik
    class_counts = defaultdict(int)
    ignored_count = 0

    # 4) Skriv bilder + labels för train/val
    def process_split(split_names, img_dir, lbl_dir, split_name):
        nonlocal next_id, ignored_count
        for img_name in split_names:
            recs = image_records[img_name]

            # Hämta bildinfo från första recordet
            base = recs[0]
            w_img = base["width"]
            h_img = base["height"]
            src_img = IMAGES_DIR / img_name
            if not src_img.exists():
                print(f"[{split_name}] Saknar bild:", src_img)
                continue

            # Kopiera bild
            dst_img = img_dir / img_name
            shutil.copy2(src_img, dst_img)

            # Labels-fil
            txt_name = Path(img_name).with_suffix(".txt")
            lbl_path = lbl_dir / txt_name

            with lbl_path.open("w", encoding="utf-8") as lf:
                for rec in recs:
                    label = rec["label"]
                    
                    # Ignorera __IGNORE__ och IGNORE
                    if label == "__IGNORE__" or label.upper() == "IGNORE":
                        ignored_count += 1
                        continue

                    bbox = rec["bbox"]  # {"x": x, "y": y, "w": w, "h": h}

                    # map label → class id
                    if label not in class_to_id:
                        class_to_id[label] = next_id
                        next_id += 1

                    cls_id = class_to_id[label]
                    class_counts[label] += 1

                    x = bbox["x"]
                    y = bbox["y"]
                    w = bbox["w"]
                    h = bbox["h"]

                    # Konvertera till YOLO-format (center x, center y, width, height - normaliserat)
                    cx = (x + w / 2) / w_img
                    cy = (y + h / 2) / h_img
                    nw = w / w_img
                    nh = h / h_img
                    
                    # Begränsa värden till giltigt intervall
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    nw = max(0.001, min(1.0, nw))
                    nh = max(0.001, min(1.0, nh))

                    lf.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    process_split(train_set, IMG_TRAIN, LBL_TRAIN, "train")
    process_split(val_set, IMG_VAL, LBL_VAL, "val")

    if not class_to_id:
        print("Inga giltiga tränings-exempel hittades.")
        return None, None

    # 5) Skriv data.yaml
    data_yaml = DATASET_ROOT / "data.yaml"
    class_list = [None] * len(class_to_id)
    for name, idx in class_to_id.items():
        class_list[idx] = name

    with data_yaml.open("w", encoding="utf-8") as f:
        f.write(f"path: {DATASET_ROOT.as_posix()}\n")
        f.write("train: images/train\n")
        if val_set:
            f.write("val: images/val\n")
        else:
            f.write("val: images/train\n")  # fallback om inga val-bilder
        f.write(f"\nnc: {len(class_list)}\n")  # antal klasser
        f.write("names:\n")
        for idx, name in enumerate(class_list):
            f.write(f"  {idx}: {name}\n")

    print("\n" + "="*50)
    print("Dataset klart!")
    print("="*50)
    print(f"Antal train-bilder: {len(train_set)}")
    print(f"Antal val-bilder: {len(val_set)}")
    print(f"Antal klasser: {len(class_list)}")
    print(f"Ignorerade annotationer: {ignored_count}")
    print("\nKlasser:", class_list)
    print("\nKlass-fördelning:")
    for cls_name, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls_name}: {cnt}")
    print("="*50 + "\n")

    return data_yaml, class_list


def train_yolo(data_yaml, prefer_run_name: str | None = None):
    """
    Tränar YOLO med optimerade inställningar för container-inspektion:
      - Startar från pti_best.pt om den finns (inkrementell träning)
      - Annars från yolov8n.pt
      - Optimerad augmentation för industriella/container-scenarier
      - Större bildstorlek för bättre detektion av små defekter
    """
    # Startmodell: försök återuppta från senaste last.pt, annars pti_best.pt, annars basmodell

    def find_latest_last_checkpoint():
        """Sök i runs/detect/train* efter senaste weights/last.pt och returnera stigen.

        Om prefer_run_name är satt, försök först med den specifika run:en.
        """
        runs_detect = ROOT / "runs" / "detect"
        if not runs_detect.exists():
            return None
        # Försök först med specifik run om angiven
        if prefer_run_name:
            preferred = runs_detect / prefer_run_name / "weights" / "last.pt"
            if preferred.exists():
                return preferred
        candidates = []
        for d in runs_detect.iterdir():
            if not d.is_dir():
                continue
            if not d.name.startswith("train"):
                continue
            last_path = d / "weights" / "last.pt"
            if last_path.exists():
                try:
                    mtime = last_path.stat().st_mtime
                except Exception:
                    mtime = 0
                candidates.append((mtime, last_path))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    resume_ckpt = find_latest_last_checkpoint()
    if resume_ckpt is not None:
        print(f"Återupptar träning från senaste checkpoint: {resume_ckpt}")
        model = YOLO(str(resume_ckpt))
        resume_flag = True
    else:
        pti_path = ROOT / "pti_best.pt"
        if pti_path.exists():
            print(f"Laddar befintlig modell: {pti_path}")
            model = YOLO(str(pti_path))
        else:
            print("Ingen pti_best.pt hittad – startar från yolov8n.pt")
            model = YOLO("yolov8n.pt")
        resume_flag = False

    print("\nStartar YOLO-träning med optimerade inställningar...")
    print(f"Data config: {data_yaml}")
    
    results = model.train(
        data=str(data_yaml),
        
        # Träningstid
        epochs=120,             # Mer träning när vi inte vill stoppa tidigt
        patience=0,             # Ingen early stopping
        
        # Bildinställningar
        imgsz=1024,             # Större input för bättre detektion av små defekter
        batch=4,                # Mindre batch för större bilder
        
        # Inlärningshastighet (lägre för finjustering)
        lr0=0.008,              # Initial learning rate
        lrf=0.01,               # Final learning rate factor
        warmup_epochs=5,        # Mer warmup för stabilitet
        
        # Augmentation - anpassad för industriell inspektion
        degrees=8.0,            # Lätt rotation (containers roteras sällan mycket)
        translate=0.15,         # Förskjutning
        scale=0.3,              # Skalvariation (för olika avstånd)
        shear=3.0,              # Lätt skjuvning
        perspective=0.0005,     # Minimal perspektivförvrängning
        flipud=0.0,             # Ingen vertikal flip (containers har definierad orientering)
        fliplr=0.5,             # Horisontell flip OK
        
        # Färgaugmentation (viktigt för smuts/skada-detektion)
        hsv_h=0.01,             # Lätt nyanvariation
        hsv_s=0.5,              # Mättnadsvariation (ljusförändringar)
        hsv_v=0.4,              # Ljusstyrkevariation
        
        # Mosaic och mixup
        mosaic=0.8,             # Mosaic-augmentation (hjälper med litet dataset)
        mixup=0.1,              # Lätt mixup
        copy_paste=0.1,         # Copy-paste augmentation för objekt
        
        # Andra inställningar
        close_mosaic=15,        # Stäng av mosaic i sista epokerna
        amp=True,               # Automatic mixed precision
        cache=True,             # Cacha bilder för snabbare träning
        workers=4,              # DataLoader workers
        
        # Förlustvikter (betona box-noggrannhet för defektdetektion)
        box=7.5,                # Box loss weight
        cls=0.5,                # Class loss weight
        dfl=1.5,                # Distribution focal loss weight
        
        # Validering
        val=True,
        plots=True,
        save=True,
        save_period=10,         # Spara checkpoint var 10:e epok
        resume=resume_flag,     # Återuppta om vi hittade last.pt
    )

    # Ultralytics sparar weights/best.pt i runs/detect/train*/weights/best.pt
    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        dst = ROOT / "pti_best.pt"
        shutil.copy2(best, dst)
        print(f"\n✓ Ny modell sparad som: {dst}")
        
        # Spara också en backup med tidsstämpel
        backup_name = f"pti_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        backup_path = ROOT / "model_backups"
        backup_path.mkdir(exist_ok=True)
        shutil.copy2(best, backup_path / backup_name)
        print(f"✓ Backup sparad som: {backup_path / backup_name}")
    else:
        print("✗ Kunde inte hitta best.pt efter träning.")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train or resume YOLO training for container inspection")
    parser.add_argument("--resume-run", type=str, default=None,
                        help="Name under runs/detect/<name> to resume from weights/last.pt (e.g., 'train2')")
    args = parser.parse_args()

    data_yaml, classes = build_dataset()
    if data_yaml is None:
        return
    print("Startar YOLO-träning...")
    train_yolo(data_yaml, prefer_run_name=args.resume_run)
    print("KLART.")


if __name__ == "__main__":
    main()
