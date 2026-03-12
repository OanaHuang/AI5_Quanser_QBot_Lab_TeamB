#-----------------------------------------------------------------------------#
#                         Step 3 - Batch Auto-Labeling Script                 #
#                                                                             #
# Before running this script, ensure you have completed:                      #
#    Step 1: dataset/raw/images/ and dataset/raw/coords.csv are generated      #
#    Step 2: dataset/raw/regions.json is saved                                #
#                                                                             #
# Run:                                                                        #
#    python data_collection/step3_auto_label.py                               #
#                                                                             #
# Output:                                                                     #
#    dataset/labeled/straight/   ← Straight-line images                       #
#    dataset/labeled/corner/     ← Corner/Turn images                         #
#    dataset/labeled/crossroad/  ← Intersection/Crossroad images                #
#    dataset/raw/coords.csv      ← 'label' column filled (updated in place)   #
#    dataset/label_report.txt    ← Labeling statistics report                 #
#-----------------------------------------------------------------------------#

import os
import csv
import json
import shutil
import numpy as np
from collections import defaultdict

# -- Path Configuration --------------------------------------------------------
DATASET_ROOT  = "dataset"
COORDS_CSV    = os.path.join(DATASET_ROOT, "raw", "coords.csv")
REGIONS_JSON  = os.path.join(DATASET_ROOT, "raw", "regions.json")
IMG_SRC_DIR   = os.path.join(DATASET_ROOT, "raw", "images")
LABELED_DIR   = os.path.join(DATASET_ROOT, "labeled")
REPORT_PATH   = os.path.join(DATASET_ROOT, "label_report.txt")

LABEL_PRIORITY = ["crossroad", "corner", "straight"]
# 'crossroad' has highest priority: if a point falls into both 'corner' and 
# 'crossroad' regions, it will be labeled as 'crossroad'.


# =============================================================================

def load_regions(json_path):
    """Load the region definitions saved in Step 2"""
    with open(json_path, "r") as f:
        regions = json.load(f)
    print(f"[Step3] Loaded {len(regions)} region definitions")
    return regions


def classify_point(x, y, regions):
    """
    Determines the label for coordinates (x, y).
    If falling into multiple regions, the highest priority from LABEL_PRIORITY is chosen.
    Returns 'straight' as a fallback if the point is not within any defined region.
    """
    matched_labels = set()
    for region in regions:
        dist = np.hypot(x - region["cx"], y - region["cy"])
        if dist <= region["r"]:
            matched_labels.add(region["label"])

    for label in LABEL_PRIORITY:
        if label in matched_labels:
            return label

    return "straight"   # Fallback: outside any annotated region -> straight


def run_labeling():
    # ---- Check files ----
    for path in [COORDS_CSV, REGIONS_JSON, IMG_SRC_DIR]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find {path}. Please run previous steps first.")

    # ---- Load ----
    regions = load_regions(REGIONS_JSON)

    # ---- Read CSV ----
    rows = []
    with open(COORDS_CSV, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    print(f"[Step3] {len(rows)} records found for labeling")

    # ---- Create output directories ----
    for label in LABEL_PRIORITY:
        os.makedirs(os.path.join(LABELED_DIR, label), exist_ok=True)

    # ---- Label each record and copy images ----
    stats = defaultdict(int)
    skipped = 0

    for row in rows:
        img_name = row["image_name"]
        src_path = os.path.join(IMG_SRC_DIR, img_name)

        if not os.path.isfile(src_path):
            skipped += 1
            continue

        try:
            x, y = float(row["x"]), float(row["y"])
        except ValueError:
            skipped += 1
            continue

        label = classify_point(x, y, regions)
        row["label"] = label

        # Copy image to the corresponding subdirectory
        dst_path = os.path.join(LABELED_DIR, label, img_name)
        shutil.copy2(src_path, dst_path)
        stats[label] += 1

    # ---- Write back to CSV (filling the label column) ----
    with open(COORDS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ---- Generate Report ----
    total = sum(stats.values())
    report_lines = [
        "=" * 50,
        "          Step 3 Auto-Labeling Report",
        "=" * 50,
        f"Total images:      {total + skipped}",
        f"Successfully labeled: {total}",
        f"Skipped (missing):    {skipped}",
        "",
        "Label Distribution:",
    ]
    for label in LABEL_PRIORITY:
        cnt  = stats[label]
        pct  = (cnt / total * 100) if total > 0 else 0
        bar  = "█" * int(pct / 2)
        report_lines.append(f"  {label:<12} {cnt:>5} pics  ({pct:5.1f}%)  {bar}")
    report_lines += [
        "",
        f"Output directory: {LABELED_DIR}",
        f"CSV Updated:      {COORDS_CSV}",
        "=" * 50,
    ]
    report_text = "\n".join(report_lines)
    print(report_text)

    with open(REPORT_PATH, "w") as f:
        f.write(report_text)
    print(f"[Step3] Report saved to {REPORT_PATH}")


# =============================================================================
if __name__ == "__main__":
    run_labeling()