"""
Main orchestration for microscopy analysis (refactored)
"""
import os
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import messagebox
import csv

from detection import launch_cell_detector
from subareas import launch_subarea_selector
from roi_selector import select_roi_and_show

# Set SKIP_CELL_DETECTION = True during development to speed up testing
SKIP_CELL_DETECTION = True


# `select_roi_and_show` moved to `roi_selector.py` to reduce main module size


if __name__ == '__main__':
    image_path = "Hipp2.1.tiff"
    selector, store = select_roi_and_show(image_path)

    if hasattr(selector, 'masked_img') and getattr(selector, 'masked_img') is not None:
        if not SKIP_CELL_DETECTION:
            launch_cell_detector(selector.masked_img, image_path)
        else:
            print("\nCell detection skipped (SKIP_CELL_DETECTION = True)")
            print("Opening subarea selector for pixel-threshold counts.")

        try:
            roi_mask_cropped = None
            try:
                if hasattr(selector, 'mask') and hasattr(selector, 'crop_offset') and selector.mask is not None:
                    rmin, cmin = selector.crop_offset
                    rmax = rmin + selector.masked_img.shape[0] - 1
                    cmax = cmin + selector.masked_img.shape[1] - 1
                    roi_mask_cropped = selector.mask[rmin:rmax+1, cmin:cmax+1]
            except Exception:
                roi_mask_cropped = None

            launch_subarea_selector(selector.masked_img, image_path, roi_mask=roi_mask_cropped, store=store)
        except Exception as e:
            print(f"Failed to launch subarea selector: {e}")
    else:
        print("No ROI selected. Exiting.")
