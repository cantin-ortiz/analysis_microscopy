"""
Cell detection (StarDist) module
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button
import os
import tkinter as tk
from tkinter import messagebox

# Module-level StarDist model cache
STARDIST_MODEL = None


def _load_stardist_model():
    """Lazily load (and cache) the StarDist 2D fluorescence model."""
    global STARDIST_MODEL
    if STARDIST_MODEL is None:
        print("Loading StarDist model (first run may download weights)...")
        from stardist.models import StarDist2D
        STARDIST_MODEL = StarDist2D.from_pretrained('2D_versatile_fluo')
        print("StarDist model '2D_versatile_fluo' loaded successfully")
    return STARDIST_MODEL


def launch_cell_detector(roi_image, image_path, store=None):
    """Launch interactive cell detection interface using StarDist."""

    # Create new figure for cell detection
    fig, ax = plt.subplots(figsize=(12, 10))
    # Try to open the figure maximized
    try:
        mgr = plt.get_current_fig_manager()
        try:
            win = mgr.window
            try:
                win.state('zoomed')
            except Exception:
                try:
                    mgr.window.showMaximized()
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass
    # Make room on the left for sliders and right for controls
    plt.subplots_adjust(left=0.15, bottom=0.25, right=0.78, top=0.95)

    # Store original image for brightness/contrast adjustments
    image_original = roi_image.copy()
    image_min, image_max = float(roi_image.min()), float(roi_image.max())

    # Load saved brightness/contrast values from store if available
    saved_brightness = store.get('brightness', 0) if store else 0
    saved_contrast = store.get('contrast', 1.0) if store else 1.0

    # StarDist parameters
    # prob_thresh: probability threshold — lower → more detections (default ~0.48)
    # nms_thresh:  non-maximum suppression overlap threshold (default 0.3)
    # scale:       rescale factor sent to StarDist (None = auto)
    # min_size:    post-filter: discard objects smaller than this many pixels
    initial_prob_thresh = 0.48
    initial_nms_thresh = 0.3
    initial_scale = 1.0
    initial_min_size = 50

    detection_params = {
        'prob_thresh': initial_prob_thresh,
        'nms_thresh': initial_nms_thresh,
        'scale': initial_scale,
        'min_size': initial_min_size,
    }

    det_image = roi_image
    det_ax = ax
    det_im = ax.imshow(roi_image, cmap='gray', vmin=image_min, vmax=image_max)
    det_circles = []
    outlines_visible = True
    detected_cells = np.empty((0, 4))

    # Brightness slider (left side)
    ax_brightness = plt.axes([0.02, 0.35, 0.02, 0.5])
    slider_brightness = Slider(
        ax_brightness, 'Brightness', -100, 100, valinit=saved_brightness,
        orientation='vertical', valstep=1
    )

    # Contrast slider (left side)
    ax_contrast = plt.axes([0.06, 0.35, 0.02, 0.5])
    slider_contrast = Slider(
        ax_contrast, 'Contrast', 0.1, 3.0, valinit=saved_contrast,
        orientation='vertical', valstep=0.05
    )

    def update_image_display(val=None):
        """Update image display based on brightness and contrast sliders"""
        try:
            brightness = slider_brightness.val
            contrast = slider_contrast.val

            if store:
                store.set('brightness', brightness)
                store.set('contrast', contrast)

            img_mid = (image_min + image_max) / 2.0
            adjusted = (image_original - img_mid) * contrast + img_mid + brightness
            adjusted = np.clip(adjusted, image_min, image_max)

            det_im.set_data(adjusted)
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating image display: {e}")

    slider_brightness.on_changed(update_image_display)
    slider_contrast.on_changed(update_image_display)

    if saved_brightness != 0 or saved_contrast != 1.0:
        update_image_display()

    # ---- StarDist detection --------------------------------------------------
    def detect_cells():
        nonlocal det_circles, detected_cells
        # Remove previous circles
        for circle in det_circles:
            circle.remove()
        det_circles = []

        model = _load_stardist_model()

        # Normalize image to [0, 1] for StarDist
        from csbdeep.utils import normalize
        img_norm = normalize(det_image, 1, 99.8)

        print("Running StarDist segmentation...")
        scale_val = detection_params['scale']
        labels, details = model.predict_instances(
            img_norm,
            prob_thresh=detection_params['prob_thresh'],
            nms_thresh=detection_params['nms_thresh'],
            scale=scale_val if scale_val != 1.0 else None,
            return_predict=False,
        )

        from skimage.measure import regionprops
        regions = regionprops(labels)

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
        det_ax.set_title(
            f'Cell Detection (StarDist): {len(filtered_cells)} cells detected - Outlines {status}',
            fontsize=14, fontweight='bold',
        )
        det_ax.axis('off')
        print(f"\nDetected {len(filtered_cells)} cells using StarDist")

    # ---- Sliders -------------------------------------------------------------
    slider_width = 0.63
    ax_prob = plt.axes([0.15, 0.15, slider_width, 0.03])
    ax_nms = plt.axes([0.15, 0.11, slider_width, 0.03])
    ax_scale = plt.axes([0.15, 0.07, slider_width, 0.03])
    ax_min_size = plt.axes([0.15, 0.03, slider_width, 0.03])

    slider_prob = Slider(ax_prob, 'Prob Threshold', 0.01, 0.99, valinit=initial_prob_thresh, valstep=0.01)
    slider_nms = Slider(ax_nms, 'NMS Threshold', 0.01, 0.99, valinit=initial_nms_thresh, valstep=0.01)
    slider_scale = Slider(ax_scale, 'Scale', 0.25, 4.0, valinit=initial_scale, valstep=0.05)
    slider_min_size = Slider(ax_min_size, 'Min Area (px²)', 0, 500, valinit=initial_min_size, valstep=5)

    # ---- Load from store? ----------------------------------------------------
    loaded_from_store = False
    try:
        if store is not None:
            try:
                existing_cells = store.get('segmented_cells', None)
            except Exception:
                existing_cells = getattr(store, 'data', {}).get('segmented_cells') if hasattr(store, 'data') else None

            if existing_cells:
                try:
                    root = tk.Tk()
                    root.withdraw()
                    try:
                        root.attributes('-topmost', True)
                    except Exception:
                        pass
                    ask = messagebox.askyesno(
                        'Load segmentation',
                        f"Found {len(existing_cells)} segmented cells in analysis file. "
                        "Load them instead of running StarDist?",
                        parent=root,
                    )
                    root.destroy()
                except Exception:
                    ask = messagebox.askyesno(
                        'Load segmentation',
                        f"Found {len(existing_cells)} segmented cells in analysis file. "
                        "Load them instead of running StarDist?",
                    )

                if ask:
                    # Restore saved slider settings if available
                    try:
                        saved_settings = store.get('cell_detection_settings', None)
                    except Exception:
                        saved_settings = getattr(store, 'data', {}).get('cell_detection_settings') if hasattr(store, 'data') else None

                    if isinstance(saved_settings, dict):
                        for k in ('prob_thresh', 'nms_thresh', 'scale', 'min_size'):
                            if k in saved_settings:
                                try:
                                    detection_params[k] = saved_settings[k]
                                except Exception:
                                    pass

                    # Apply settings to sliders
                    try:
                        slider_prob.set_val(detection_params['prob_thresh'])
                        slider_nms.set_val(detection_params['nms_thresh'])
                        slider_scale.set_val(detection_params['scale'])
                        slider_min_size.set_val(detection_params['min_size'])
                    except Exception:
                        pass

                    # Populate detected_cells from store
                    try:
                        parsed = []
                        for item in existing_cells:
                            x = float(item.get('x', item.get('cx', 0)))
                            y = float(item.get('y', item.get('cy', 0)))
                            r = float(item.get('radius', item.get('r', 0)))
                            area = float(item.get('area', 0))
                            parsed.append((y, x, r, area))
                        detected_cells = np.array(parsed) if len(parsed) > 0 else np.empty((0, 4))
                    except Exception:
                        detected_cells = np.empty((0, 4))

                    # Render loaded detections (no StarDist run)
                    def render_loaded_cells():
                        nonlocal det_circles
                        for c in det_circles:
                            try:
                                c.remove()
                            except Exception:
                                pass
                        det_circles = []
                        for y, x, r, area in (detected_cells if detected_cells.size else []):
                            try:
                                circle = Circle((x, y), r, fill=False, edgecolor='red', linewidth=2, alpha=0.8)
                                det_ax.add_patch(circle)
                                det_circles.append(circle)
                            except Exception:
                                pass
                        for circle in det_circles:
                            circle.set_visible(outlines_visible)
                        status = "VISIBLE" if outlines_visible else "HIDDEN"
                        det_ax.set_title(
                            f'Cell Detection (Loaded): {len(det_circles)} cells - Outlines {status}',
                            fontsize=14, fontweight='bold',
                        )
                        det_ax.axis('off')
                        try:
                            det_ax.figure.canvas.draw_idle()
                        except Exception:
                            pass
                        print(f"\nLoaded {len(det_circles)} cells from AnalysisStore")

                    render_loaded_cells()
                    loaded_from_store = True
    except Exception:
        loaded_from_store = False

    # ---- Buttons -------------------------------------------------------------
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

        detection_params['prob_thresh'] = slider_prob.val
        detection_params['nms_thresh'] = slider_nms.val
        detection_params['scale'] = slider_scale.val
        detection_params['min_size'] = slider_min_size.val

        detect_cells()

        btn_detect.label.set_text('Recompute')
        btn_detect.color = '0.85'
        btn_detect.hovercolor = '0.95'
        plt.draw()

    def save_cell_detection(event):
        nonlocal detected_cells, detection_params
        store_obj = store
        if len(detected_cells) == 0:
            print("No cells detected to save")
            return
        if store_obj is None:
            try:
                from analysis_store import AnalysisStore
                store_obj = AnalysisStore(image_path)
            except Exception as e:
                raise RuntimeError(f"Unable to create AnalysisStore for '{image_path}': {e}")

        try:
            cells_payload = [
                {
                    'cell_id': int(idx),
                    'x': float(cell[1]),
                    'y': float(cell[0]),
                    'radius': float(cell[2]),
                    'area': float(cell[3]),
                }
                for idx, cell in enumerate(detected_cells, 1)
            ]

            try:
                existing = store_obj.get('segmented_cells', None)
            except Exception:
                existing = getattr(store_obj, 'data', {}).get('segmented_cells') if hasattr(store_obj, 'data') else None

            if existing:
                try:
                    root_confirm = tk.Tk()
                    root_confirm.withdraw()
                    try:
                        root_confirm.attributes('-topmost', True)
                    except Exception:
                        pass
                    overwrite = messagebox.askyesno(
                        'Overwrite segmentation',
                        f"Analysis JSON already contains {len(existing)} segmented cells. Overwrite them?",
                        parent=root_confirm,
                    )
                    try:
                        root_confirm.destroy()
                    except Exception:
                        pass
                except Exception:
                    overwrite = messagebox.askyesno(
                        'Overwrite segmentation',
                        f"Analysis JSON already contains {len(existing)} segmented cells. Overwrite them?",
                    )

                if not overwrite:
                    try:
                        root_info = tk.Tk()
                        root_info.withdraw()
                        try:
                            root_info.attributes('-topmost', True)
                        except Exception:
                            pass
                        messagebox.showinfo(
                            'Save Cancelled',
                            'Existing segmentation preserved; save cancelled. Moving to subarea selection.',
                            parent=root_info,
                        )
                        try:
                            root_info.destroy()
                        except Exception:
                            pass
                    except Exception:
                        try:
                            messagebox.showinfo(
                                'Save Cancelled',
                                'Existing segmentation preserved; save cancelled. Moving to subarea selection.',
                            )
                        except Exception:
                            pass

                    try:
                        import matplotlib.pyplot as _plt
                        try:
                            _plt.close(fig)
                        except Exception:
                            pass
                    except Exception:
                        pass
                    return

            # Write results
            try:
                store_obj.set('segmented_cells', cells_payload)
                store_obj.set('cell_detection_settings', detection_params)
            except Exception:
                raise
        except Exception as e:
            raise RuntimeError(f"Failed to persist detection results to AnalysisStore: {e}")

        saved_to = getattr(store_obj, 'filename', None)

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        messagebox.showinfo(
            "Save Complete",
            f"Cell detection results saved into analysis JSON:\n{os.path.basename(saved_to)}\n\n"
            f"Total cells detected: {len(detected_cells)}",
            parent=root,
        )
        try:
            root.destroy()
        except Exception:
            pass

        try:
            import matplotlib.pyplot as _plt
            try:
                _plt.close(fig)
            except Exception:
                pass
        except Exception:
            pass

    # Help text
    det_help_text = "H: Toggle outlines visibility"
    help_text_artist = det_ax.text(
        0.02, 0.98, det_help_text, transform=det_ax.transAxes,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
    )

    def on_key_det(event):
        nonlocal outlines_visible
        if event.key == 'h':
            outlines_visible = not outlines_visible
            for circle in det_circles:
                circle.set_visible(outlines_visible)
            status = "VISIBLE" if outlines_visible else "HIDDEN"
            det_ax.set_title(
                f'Cell Detection (StarDist): {len(detected_cells)} cells detected - Outlines {status}',
                fontsize=14, fontweight='bold',
            )
            det_ax.figure.canvas.draw_idle()

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

    # Initial detection (skip if loaded from store)
    if not loaded_from_store:
        detect_cells()

    fig.canvas.draw()
    plt.show()

    return
