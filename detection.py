"""
Cell detection (Cellpose) module
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
import csv
import os
import tkinter as tk
from tkinter import messagebox

# Module-level CELLPOSE model cache
CELLPOSE_MODEL = None


def launch_cell_detector(roi_image, image_path):
    """Launch interactive cell detection interface (module-level)"""
    global CELLPOSE_MODEL

    # Create new figure for cell detection
    fig, ax = plt.subplots(figsize=(12, 10))
    # Make room on the right for controls so buttons don't overlap the image
    plt.subplots_adjust(bottom=0.25, right=0.78, top=0.95)

    # Initial parameters for Cellpose
    initial_diameter = 30
    initial_flow_threshold = 0.4
    initial_cellprob_threshold = 0.0
    initial_min_size = 100

    detection_params = {
        'diameter': initial_diameter,
        'flow_threshold': initial_flow_threshold,
        'cellprob_threshold': initial_cellprob_threshold,
        'min_size': initial_min_size
    }

    det_image = roi_image
    det_ax = ax
    det_im = ax.imshow(roi_image, cmap='gray')
    det_circles = []
    outlines_visible = True
    detected_cells = np.empty((0, 4))

    def detect_cells():
        nonlocal det_circles, detected_cells
        # Remove previous circles
        for circle in det_circles:
            circle.remove()
        det_circles = []

        # Initialize Cellpose model (import lazily to avoid slowing script startup)
        global CELLPOSE_MODEL
        if CELLPOSE_MODEL is None:
            print("Loading Cellpose model (first run may take a moment)...")
            from cellpose import models
            CELLPOSE_MODEL = models.CellposeModel(gpu=True)
            print(f"Cellpose will use GPU: {CELLPOSE_MODEL.gpu}")

        print("Running Cellpose segmentation (this may take 30-60 seconds on CPU)...")
        result = CELLPOSE_MODEL.eval(
            det_image,
            diameter=detection_params['diameter'],
            flow_threshold=detection_params['flow_threshold'],
            cellprob_threshold=detection_params['cellprob_threshold']
        )

        if len(result) == 4:
            masks, flows, styles, diams = result
        else:
            masks = result[0]

        from skimage.measure import regionprops
        regions = regionprops(masks)

        min_area = detection_params['min_size']
        filtered_cells = []

        for region in regions:
            if region.area >= min_area:
                y, x = region.centroid
                r = np.sqrt(region.area / np.pi)
                filtered_cells.append((y, x, r, region.area))

        detected_cells = np.array(filtered_cells) if len(filtered_cells) > 0 else np.empty((0, 4))

        for y, x, r, area in filtered_cells:
            circle = Circle((x, y), r, fill=False, edgecolor='red', linewidth=2, alpha=0.8)
            det_ax.add_patch(circle)
            det_circles.append(circle)

        for circle in det_circles:
            circle.set_visible(outlines_visible)

        status = "VISIBLE" if outlines_visible else "HIDDEN"
        det_ax.set_title(f'Cell Detection (Cellpose): {len(filtered_cells)} cells detected - Outlines {status}', 
                         fontsize=14, fontweight='bold')
        det_ax.axis('off')
        print(f"\nDetected {len(filtered_cells)} cells using Cellpose")

    # Create sliders (shrink width to respect right margin)
    slider_width = 0.63  # 0.78 - 0.15
    ax_diameter = plt.axes([0.15, 0.15, slider_width, 0.03])
    ax_flow = plt.axes([0.15, 0.11, slider_width, 0.03])
    ax_cellprob = plt.axes([0.15, 0.07, slider_width, 0.03])
    ax_min_size = plt.axes([0.15, 0.03, slider_width, 0.03])

    slider_diameter = Slider(ax_diameter, 'Cell Diameter (px)', 5, 100, valinit=initial_diameter, valstep=1)
    slider_flow = Slider(ax_flow, 'Flow Threshold', 0.1, 2.0, valinit=initial_flow_threshold, valstep=0.05)
    slider_cellprob = Slider(ax_cellprob, 'Cell Prob Threshold', -6, 6, valinit=initial_cellprob_threshold, valstep=0.5)
    slider_min_size = Slider(ax_min_size, 'Min Area (px²)', 10, 1000, valinit=initial_min_size, valstep=10)

    # Place control buttons stacked in the right margin so they don't overlap each other
    ax_detect = plt.axes([0.80, 0.90, 0.15, 0.05])
    ax_help = plt.axes([0.80, 0.82, 0.15, 0.05])
    ax_save = plt.axes([0.80, 0.74, 0.15, 0.05])
    btn_detect = Button(ax_detect, 'Recompute')
    btn_help = Button(ax_help, 'Hide Help')
    btn_save = Button(ax_save, 'Save Results')

    def update_detection(val):
        nonlocal detection_params
        btn_detect.label.set_text('Computing...')
        btn_detect.color = 'orange'
        btn_detect.hovercolor = 'orange'
        btn_detect.ax.figure.canvas.draw()
        btn_detect.ax.figure.canvas.flush_events()
        plt.pause(0.1)

        detection_params['diameter'] = slider_diameter.val
        detection_params['flow_threshold'] = slider_flow.val
        detection_params['cellprob_threshold'] = slider_cellprob.val
        detection_params['min_size'] = slider_min_size.val

        detect_cells()

        btn_detect.label.set_text('Recompute')
        btn_detect.color = '0.85'
        btn_detect.hovercolor = '0.95'
        plt.draw()

    def save_cell_detection(event):
        nonlocal detected_cells, detection_params
        if len(detected_cells) == 0:
            print("No cells detected to save")
            return
        base_name = os.path.splitext(image_path)[0]
        csv_path = f"{base_name}_cells.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cell_id', 'x', 'y', 'radius', 'area'])
            for idx, cell in enumerate(detected_cells, 1):
                y, x, r, area = cell
                writer.writerow([idx, x, y, r, area])

        settings_csv_path = f"{base_name}_settings.csv"
        with open(settings_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['parameter', 'value'])
            for param, value in detection_params.items():
                writer.writerow([param, value])

        print(f"\nCell detection results saved to: {csv_path}")
        print(f"Cellpose settings saved to: {settings_csv_path}")

        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "Save Complete",
            f"Files saved successfully!\n\n📊 Cell data: {os.path.basename(csv_path)}\n⚙️ Settings: {os.path.basename(settings_csv_path)}\n\nTotal cells detected: {len(detected_cells)}",
            parent=None
        )
        root.destroy()

    # Small help text in the detector view (top-left of image axes)
    det_help_text = (
        "H: Toggle outlines visibility"
    )
    help_text_artist = det_ax.text(0.02, 0.98, det_help_text, transform=det_ax.transAxes,
                                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def on_key_det(event):
        nonlocal outlines_visible
        if event.key == 'h':
            outlines_visible = not outlines_visible
            for circle in det_circles:
                circle.set_visible(outlines_visible)
            status = "VISIBLE" if outlines_visible else "HIDDEN"
            det_ax.set_title(f'Cell Detection (Cellpose): {len(detected_cells)} cells detected - Outlines {status}', fontsize=14, fontweight='bold')
            det_ax.figure.canvas.draw_idle()

    # Toggle help when clicking the help button
    def _toggle_help_det(event):
        try:
            vis = help_text_artist.get_visible()
            help_text_artist.set_visible(not vis)
            btn_help.label.set_text('Show Help' if vis else 'Hide Help')
            det_ax.figure.canvas.draw_idle()
        except Exception:
            pass

    btn_help.on_clicked(_toggle_help_det)

    btn_detect.on_clicked(update_detection)
    btn_save.on_clicked(save_cell_detection)
    cid_key_det = fig.canvas.mpl_connect('key_press_event', on_key_det)

    # Initial detection
    detect_cells()

    # Draw canvas explicitly to preserve our manual layout (avoid tight_layout)
    fig.canvas.draw()
    plt.show()
