"""
Cell detection (StarDist) module
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button, RangeSlider
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


def launch_cell_detector(roi_image, image_path, store=None, um2_per_pixel=None):
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
    plt.subplots_adjust(left=0.15, bottom=0.30, right=0.78, top=0.95)

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
    initial_prob_thresh = 0.48
    initial_nms_thresh = 0.3
    initial_scale = 1.0

    # Area / fluorescence display units
    if um2_per_pixel is not None and um2_per_pixel > 0:
        area_factor = um2_per_pixel   # px² × area_factor → µm²
        area_unit = 'µm²'
    else:
        area_factor = 1.0
        area_unit = 'px²'

    # Area range defaults (in pixels)
    initial_area_min_px = 50
    initial_area_max_px = 5000
    # Fluorescence range defaults (image intensity)
    initial_fluor_min = 0.0
    initial_fluor_max = float(roi_image.max())

    detection_params = {
        'prob_thresh': initial_prob_thresh,
        'nms_thresh': initial_nms_thresh,
        'scale': initial_scale,
        'area_range_px': [initial_area_min_px, initial_area_max_px],
        'fluor_range': [initial_fluor_min, initial_fluor_max],
    }

    det_image = roi_image
    det_ax = ax
    det_im = ax.imshow(roi_image, cmap='gray', vmin=image_min, vmax=image_max)
    all_circles = []   # one Circle patch per entry in all_detected, always in axes
    outlines_visible = True
    detected_cells = np.empty((0, 5))   # (y, x, r, area_px, mean_intensity) — filtered subset
    all_detected = []                    # all regions before area/fluor filtering

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

    # ---- StarDist detection + filtering --------------------------------------
    def apply_filters():
        """Instantly show/hide existing circles based on area and fluorescence sliders.
        No patches are added or removed — only set_visible() is called."""
        nonlocal detected_cells

        # Area range (display units → pixels)
        area_lo_disp, area_hi_disp = slider_area.val
        area_lo_px = area_lo_disp / area_factor
        area_hi_px = area_hi_disp / area_factor
        # Fluorescence range
        fluor_lo, fluor_hi = slider_fluor.val

        filtered = []
        for circle, (y, x, r, area_px, mean_int) in zip(all_circles, all_detected):
            passes = (area_lo_px <= area_px <= area_hi_px
                      and fluor_lo <= mean_int <= fluor_hi)
            circle._in_filter = passes
            circle.set_visible(outlines_visible and passes)
            if passes:
                filtered.append((y, x, r, area_px, mean_int))

        detected_cells = np.array(filtered) if filtered else np.empty((0, 5))

        status = "VISIBLE" if outlines_visible else "HIDDEN"
        det_ax.set_title(
            f'Cell Detection (StarDist): {len(filtered)}/{len(all_detected)} cells - Outlines {status}',
            fontsize=14, fontweight='bold',
        )
        det_ax.axis('off')
        fig.canvas.draw_idle()

    def detect_cells():
        """Run StarDist segmentation, rebuild all_circles once, then filter."""
        nonlocal all_detected, all_circles

        # Remove circles from any previous run
        for c in all_circles:
            c.remove()
        all_circles = []

        model = _load_stardist_model()

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
        regions = regionprops(labels, intensity_image=det_image)

        all_detected = []
        for region in regions:
            y, x = region.centroid
            r = np.sqrt(region.area / np.pi)
            all_detected.append((y, x, r, float(region.area), float(region.mean_intensity)))

        # Create one Circle patch per detected cell.
        # Visibility is toggled by apply_filters() — no patches are ever removed during filtering.
        for y, x, r, _, _ in all_detected:
            circle = Circle((x, y), r, fill=False, edgecolor='red', linewidth=2, alpha=0.8)
            det_ax.add_patch(circle)
            all_circles.append(circle)

        apply_filters()
        n_filt = len(detected_cells)
        print(f"\nStarDist: {len(all_detected)} objects, {n_filt} after filtering")

    # ---- Sliders -------------------------------------------------------------
    slider_width = 0.63
    ax_prob  = plt.axes([0.15, 0.23, slider_width, 0.03])
    ax_nms   = plt.axes([0.15, 0.19, slider_width, 0.03])
    ax_scale = plt.axes([0.15, 0.15, slider_width, 0.03])
    ax_area  = plt.axes([0.15, 0.08, slider_width, 0.03])
    ax_fluor = plt.axes([0.15, 0.03, slider_width, 0.03])

    slider_prob = Slider(ax_prob, 'Prob Threshold', 0.01, 0.99, valinit=initial_prob_thresh, valstep=0.01)
    slider_nms = Slider(ax_nms, 'NMS Threshold', 0.01, 0.99, valinit=initial_nms_thresh, valstep=0.01)
    slider_scale = Slider(ax_scale, 'Scale', 0.25, 4.0, valinit=initial_scale, valstep=0.05)
    slider_area = RangeSlider(
        ax_area, f'Area ({area_unit})',
        0, initial_area_max_px * area_factor,
        valinit=(initial_area_min_px * area_factor, initial_area_max_px * area_factor),
    )
    slider_fluor = RangeSlider(
        ax_fluor, 'Fluorescence',
        initial_fluor_min, initial_fluor_max,
        valinit=(initial_fluor_min, initial_fluor_max),
    )
    slider_area.on_changed(lambda val: apply_filters())
    slider_fluor.on_changed(lambda val: apply_filters())

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
                        for k in ('prob_thresh', 'nms_thresh', 'scale'):
                            if k in saved_settings:
                                try:
                                    detection_params[k] = saved_settings[k]
                                except Exception:
                                    pass
                        if 'area_range_px' in saved_settings:
                            detection_params['area_range_px'] = saved_settings['area_range_px']
                        elif 'min_size' in saved_settings:
                            detection_params['area_range_px'][0] = saved_settings['min_size']
                        if 'fluor_range' in saved_settings:
                            detection_params['fluor_range'] = saved_settings['fluor_range']

                    # Apply settings to sliders
                    try:
                        slider_prob.set_val(detection_params['prob_thresh'])
                        slider_nms.set_val(detection_params['nms_thresh'])
                        slider_scale.set_val(detection_params['scale'])
                        ar = detection_params['area_range_px']
                        slider_area.set_val((ar[0] * area_factor, ar[1] * area_factor))
                        fr = detection_params['fluor_range']
                        slider_fluor.set_val((fr[0], fr[1]))
                    except Exception:
                        pass

                    # Populate detected_cells and all_detected from store
                    try:
                        parsed = []
                        for item in existing_cells:
                            x = float(item.get('x', item.get('cx', 0)))
                            y = float(item.get('y', item.get('cy', 0)))
                            r = float(item.get('radius', item.get('r', 0)))
                            area = float(item.get('area', 0))
                            mean_int = float(item.get('mean_fluorescence', 0))
                            if mean_int == 0:
                                iy, ix = int(round(y)), int(round(x))
                                if 0 <= iy < det_image.shape[0] and 0 <= ix < det_image.shape[1]:
                                    mean_int = float(det_image[iy, ix])
                            parsed.append((y, x, r, area, mean_int))
                        all_detected = parsed.copy()
                        detected_cells = np.array(parsed) if len(parsed) > 0 else np.empty((0, 5))
                    except Exception:
                        detected_cells = np.empty((0, 5))

                    # Render loaded detections (no StarDist run)
                    def render_loaded_cells():
                        nonlocal all_circles
                        for c in all_circles:
                            try:
                                c.remove()
                            except Exception:
                                pass
                        all_circles = []
                        for row in (all_detected if all_detected else []):
                            try:
                                y, x, r = float(row[0]), float(row[1]), float(row[2])
                                circle = Circle((x, y), r, fill=False, edgecolor='red', linewidth=2, alpha=0.8)
                                det_ax.add_patch(circle)
                                all_circles.append(circle)
                            except Exception:
                                pass
                        apply_filters()
                        try:
                            det_ax.figure.canvas.draw_idle()
                        except Exception:
                            pass
                        print(f"\nLoaded {len(all_detected)} cells from AnalysisStore "
                              f"({len(detected_cells)} pass current filters)")

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
        a_lo, a_hi = slider_area.val
        detection_params['area_range_px'] = [a_lo / area_factor, a_hi / area_factor]
        detection_params['fluor_range'] = list(slider_fluor.val)

        detect_cells()

        btn_detect.label.set_text('Recompute')
        btn_detect.color = '0.85'
        btn_detect.hovercolor = '0.95'
        plt.draw()

    def save_cell_detection(event):
        nonlocal detected_cells, detection_params
        # Sync detection_params with current slider values
        detection_params['prob_thresh'] = slider_prob.val
        detection_params['nms_thresh'] = slider_nms.val
        detection_params['scale'] = slider_scale.val
        _alo, _ahi = slider_area.val
        detection_params['area_range_px'] = [_alo / area_factor, _ahi / area_factor]
        detection_params['fluor_range'] = list(slider_fluor.val)

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
                    'area_um2': float(cell[3]) * area_factor if area_factor != 1.0 else None,
                    'mean_fluorescence': float(cell[4]),
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
            for circle in all_circles:
                circle.set_visible(outlines_visible and getattr(circle, '_in_filter', True))
            n = int(detected_cells.shape[0]) if detected_cells.size else 0
            status = "VISIBLE" if outlines_visible else "HIDDEN"
            det_ax.set_title(
                f'Cell Detection (StarDist): {n}/{len(all_detected)} cells - Outlines {status}',
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
