"""
Main orchestration for microscopy analysis (refactored)
"""
import os
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import csv
import sys
from startup import choose_image_file

from detection import launch_cell_detector
from subareas import launch_subarea_selector
from roi_selector import select_roi_and_show

# Set SKIP_CELL_DETECTION = True during development to speed up testing
SKIP_CELL_DETECTION = True


# `select_roi_and_show` moved to `roi_selector.py` to reduce main module size


if __name__ == '__main__':
    def run_pipeline(initial_image=None):
        image_path = initial_image

        while True:
            if not image_path:
                image_path = choose_image_file(initialfile='Hipp2.1.tiff')
                if not image_path:
                    print('No image selected. Exiting.')
                    return

            # Open ROI selector; it may return ('change_file', path)
            result = select_roi_and_show(image_path)
            if isinstance(result, tuple) and len(result) >= 1 and result[0] == 'change_file':
                image_path = result[1]
                continue

            selector, store = result

            if not (hasattr(selector, 'masked_img') and getattr(selector, 'masked_img') is not None):
                print('No ROI selected. Returning to file selection.')
                image_path = None
                continue

            # Segmentation (optional)
            if not SKIP_CELL_DETECTION:
                seg_result = launch_cell_detector(selector.masked_img, image_path)
                if isinstance(seg_result, tuple) and seg_result and seg_result[0] == 'change_file':
                    image_path = seg_result[1]
                    continue
            else:
                print("\nCell detection skipped (SKIP_CELL_DETECTION = True)")
                print("Opening subarea selector for pixel-threshold counts.")

            # Prepare ROI mask cropped for subarea selector
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

                sub_result = launch_subarea_selector(selector.masked_img, image_path, roi_mask=roi_mask_cropped, store=store)
                if isinstance(sub_result, tuple) and sub_result and sub_result[0] == 'change_file':
                    image_path = sub_result[1]
                    continue
            except Exception as e:
                print(f"Failed to launch subarea selector: {e}")

            # Completed pipeline for this image
            return

    run_pipeline()
