#-----------------------------------------------------------------------------#
#                         Step 1 - Data Collection Script                     #
#    Based on the original lab3 main loop, synchronously saves camera images  #
#    and QBot world coordinates to a CSV file.                                #
#                                                                             #
# How to use:                                                                 #
#   1. Place this file in the same directory as lab3_line_follow.py           #
#   2. Change the setup(...) in your main program to: hQBot = setup(...)      #
#   3. Import and call DataCollector from this file inside the main loop      #
#   4. After running, images/ and coords.csv will be in dataset/raw/          #
#-----------------------------------------------------------------------------#

import os
import csv
import cv2
import time
import numpy as np

class DataCollector:
    """
    Collects data while running: saves an image every N frames and 
    records the current QBot world coordinates.
    All data is stored under <project_root>/dataset/raw/.
    """

    def __init__(self, save_every_n_frames=5, dataset_root="dataset"):
        # ---- Directory Structure ----
        self.img_dir   = os.path.join(dataset_root, "raw", "images")
        self.csv_path  = os.path.join(dataset_root, "raw", "coords.csv")
        os.makedirs(self.img_dir, exist_ok=True)

        # ---- CSV File ----
        # Create and write header if file doesn't exist; append if it does (for multiple runs)
        file_exists = os.path.isfile(self.csv_path)
        self._csv_file = open(self.csv_path, "a", newline="")
        self._writer   = csv.writer(self._csv_file)
        if not file_exists:
            self._writer.writerow(["image_name", "x", "y", "z",
                                   "heading_rad", "label"])
            # Label column is left empty, to be filled in bulk by step 3

        # ---- Counters ----
        self.frame_count  = 0        # Total frame count
        self.save_every   = save_every_n_frames
        self.image_index  = self._last_saved_index() + 1

        print(f"[DataCollector] Images saved to: {self.img_dir}")
        print(f"[DataCollector] Coordinates saved to: {self.csv_path}")
        print(f"[DataCollector] Saving every {self.save_every} frames")

    # ------------------------------------------------------------------
    def record(self, image_bgr, location, heading_rad=0.0):
        """
        Called once per frame in the main loop.
        
        Parameters
        ----------
        image_bgr  : np.ndarray  Camera image (BGR, cv2 format)
        location   : list[float] [x, y, z] from hQBot.command_and_request_state()
        heading_rad: float       Yaw angle (radians), optional
        
        Returns
        -------
        saved : bool  Whether an image was actually saved this frame
        """
        self.frame_count += 1

        if self.frame_count % self.save_every != 0:
            return False

        # Filename: 6-digit zero-padded index
        img_name = f"{self.image_index:06d}.png"
        img_path = os.path.join(self.img_dir, img_name)

        cv2.imwrite(img_path, image_bgr)

        self._writer.writerow([
            img_name,
            round(location[0], 4),
            round(location[1], 4),
            round(location[2], 4),
            round(heading_rad,  4),
            ""          # Leave label empty for now
        ])
        self._csv_file.flush()   # Prevent data loss if program crashes

        self.image_index += 1
        return True

    # ------------------------------------------------------------------
    def close(self):
        """Call at program exit to close the CSV file."""
        self._csv_file.close()
        print(f"[DataCollector] Saved {self.image_index - 1} images. Data collection finished.")

    # ------------------------------------------------------------------
    def _last_saved_index(self):
        """Reads the last index from the CSV to support resuming after a break."""
        if not os.path.isfile(self.csv_path):
            return 0
        last = 0
        with open(self.csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    idx = int(row["image_name"].split(".")[0])
                    last = max(last, idx)
                except Exception:
                    pass
        return last


# =============================================================================
# Integration Instructions for your lab3 main program:
# =============================================================================
#
#  [Change 1] Capture the return value of setup():
#      Original: setup(locationQBotP=[-1.35, 0.3, 0.05], ...)
#      Modified: hQBot = setup(locationQBotP=[-1.35, 0.3, 0.05], ...)
#
#  [Change 2] Add this line in the initialization section:
#      from data_collection.step1_collect import DataCollector
#      collector = DataCollector(save_every_n_frames=5)
#
#  [Change 3] In the main loop (if newDownCam: block), after image processing:
#
#      # --- Read QBot World Coordinates ---
#      status, location, _, _, _, _, _, _, heading, _, _ = \
#          hQBot.command_and_request_state(0, 0)
#      if status:
#          collector.record(gray_sm, location, heading)
#
#  [Change 4] In the finally: block:
#      collector.close()
#
# =============================================================================