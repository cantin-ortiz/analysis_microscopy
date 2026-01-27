"""
ROI selector helper moved out of `main.py`.
"""
import os
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import messagebox
import csv

from interactive import InteractivePolygon
from matplotlib.widgets import Button


def select_roi_and_show(image_path="Hipp2.1.tiff"):
    """Load image, handle existing polygon prompt, show main ROI selector and return the selector."""
    image = io.imread(image_path)

    # Handle multi-channel or z-stack images
    if image.ndim > 2:
        print(f"Original image shape: {image.shape}")
        if image.ndim == 3:
            # If it's RGB/multi-channel (H, W, C), convert to grayscale
            if image.shape[2] <= 4:  # Likely (height, width, channels)
                # Convert RGB to grayscale
                if image.shape[2] == 3:
                    image = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(image.dtype)
                else:
                    image = image[:, :, 0]  # Take first channel
                print(f"Converted to grayscale, shape: {image.shape}")
            else:
                # It's a z-stack (Z, H, W), use max projection
                image = np.max(image, axis=0)
                print(f"Using max projection, shape: {image.shape}")

    # Display basic information about the image
    print(f"\nImage shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Value range: [{image.min()}, {image.max()}]")

    # Check if polygon CSV exists
    base_name = os.path.splitext(image_path)[0]
    polygon_csv_path = f"{base_name}_polygon.csv"
    initial_vertices = None

    if os.path.exists(polygon_csv_path):
        root = tk.Tk()
        root.withdraw()
        response = messagebox.askyesno(
            "Existing Polygon Found",
            f"Found existing polygon file:\n{os.path.basename(polygon_csv_path)}\n\n"
            "Do you want to load it?\n\n"
            "Yes = Load existing polygon\n"
            "No = Start from scratch",
            icon='question'
        )
        root.destroy()

        if response:
            try:
                with open(polygon_csv_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)
                    initial_vertices = [[float(row[0]), float(row[1])] for row in reader]
                print(f"Loaded {len(initial_vertices)} polygon points from file")
            except Exception as e:
                print(f"Error loading polygon file: {e}")
                print("Starting with empty polygon")
                initial_vertices = None
        else:
            print("Starting with empty polygon (existing file will be overwritten when you save)")

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image, cmap='gray')
    title = 'Draw polygon ROI - See instructions in top-left corner'
    if initial_vertices:
        title += ' (LOADED EXISTING POLYGON)'
    ax.set_title(title, fontweight='bold')
    ax.axis('off')

    plt.subplots_adjust(right=0.82, top=0.95)

    selector = InteractivePolygon(ax, image, image_path, initial_vertices, ignore_enter=True)

    btn_help_ax = plt.axes([0.84, 0.92, 0.12, 0.05])
    btn_help = Button(btn_help_ax, 'Hide Help')
    ax_extract = plt.axes([0.84, 0.84, 0.12, 0.05])
    btn_extract = Button(ax_extract, 'Extract ROI')

    def _toggle_help(event):
        try:
            visible = selector.text.get_visible()
            selector.text.set_visible(not visible)
            btn_help.label.set_text('Show Help' if visible else 'Hide Help')
            selector.ax.figure.canvas.draw_idle()
        except Exception:
            pass

    def _extract_roi(event):
        try:
            if len(selector.vertices) < 3:
                print('Need at least 3 points to extract ROI')
                return
            selector.extract_roi()
        except Exception as e:
            print(f'Error extracting ROI: {e}')

    btn_help.on_clicked(_toggle_help)
    btn_extract.on_clicked(_extract_roi)

    fig.canvas.draw()
    plt.show()

    return selector
