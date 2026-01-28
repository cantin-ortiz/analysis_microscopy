"""
ROI selector helper moved out of `main.py`.
"""
import os
import numpy as np
from skimage import io
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import messagebox
import sys

from interactive import InteractivePolygon
from analysis_store import AnalysisStore
from matplotlib.widgets import Button, CheckButtons
from startup import choose_image_file


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

    # Instantiate analysis store for this image
    store = AnalysisStore(image_path)

    # Check for existing polygon in analysis JSON
    initial_vertices = None

    stored_polygon = store.get('roi_polygon', [])
    if stored_polygon:
        root = tk.Tk()
        root.withdraw()
        response = messagebox.askyesno(
            "Existing Polygon Found",
            f"Found existing polygon in analysis file:\n{os.path.basename(store.filename)}\n\n"
            "Do you want to load it?\n\n"
            "Yes = Load existing polygon\n"
            "No = Start from scratch",
            icon='question'
        )
        root.destroy()

        if response:
            try:
                # Make a deep copy so edits to the selector don't mutate the stored list
                initial_vertices = [[float(p[0]), float(p[1])] for p in stored_polygon]
                print(f"Loaded {len(initial_vertices)} polygon points from analysis JSON")
            except Exception as e:
                print(f"Error loading polygon from analysis file: {e}")
                print("Starting with empty polygon")
                initial_vertices = None
        else:
            print("Starting with empty polygon (existing analysis polygon will be overwritten when you save)")
    # No CSV fallback: polygon coordinates are stored in the analysis JSON only

    fig, ax = plt.subplots(figsize=(12, 10))
    # Try to open the figure fullscreen / maximized where possible
    try:
        mgr = plt.get_current_fig_manager()
        try:
            win = mgr.window
            try:
                win.attributes('-fullscreen', True)
            except Exception:
                try:
                    win.state('zoomed')
                except Exception:
                    try:
                        mgr.window.showMaximized()
                    except Exception:
                        try:
                            mgr.full_screen_toggle()
                        except Exception:
                            pass
        except Exception:
            try:
                mgr.full_screen_toggle()
            except Exception:
                pass
    except Exception:
        pass
    ax.imshow(image, cmap='gray')
    title = 'Draw polygon ROI - See instructions in top-left corner'
    if initial_vertices:
        title += ' (LOADED EXISTING POLYGON)'
    ax.set_title(title, fontweight='bold')
    ax.axis('off')

    plt.subplots_adjust(right=0.82, top=0.95)

    def _on_extract_callback(sel):
        # Save to analysis JSON. Only prompt if a non-empty stored polygon exists
        # and differs from the newly selected polygon.
        def polygons_equal(a, b, tol=1e-6):
            try:
                a_arr = np.array(a, dtype=float)
                b_arr = np.array(b, dtype=float)
            except Exception:
                return False
            if a_arr.shape != b_arr.shape:
                return False
            return np.allclose(a_arr, b_arr, atol=tol, rtol=0)

        existing = store.get('roi_polygon', [])
        try:
            is_equal = polygons_equal(existing, sel.vertices)
        except Exception as e:
            is_equal = False
            print(f"Error comparing polygons: {e}")
        print(f"Existing polygon points: {len(existing)}; New polygon points: {len(sel.vertices)}; equal={is_equal}")
        need_confirm = bool(existing) and not is_equal

        if need_confirm:
            root = tk.Tk()
            root.withdraw()
            resp = messagebox.askyesno(
                "Validate ROI",
                "Saving the ROI will overwrite any former outline in the .json file. Continue?",
                icon='question'
            )
            root.destroy()
            if not resp:
                print("ROI not saved (validation declined); exiting")
                try:
                    import matplotlib.pyplot as _plt
                    _plt.close(sel.ax.figure)
                except Exception:
                    pass
                sys.exit(0)

        try:
            store.set('roi_polygon', sel.vertices)
            print(f"Saved ROI polygon ({len(sel.vertices)} points) to: {store.filename}")
        except Exception as e:
            print(f"Failed to save ROI to analysis file: {e}")

    selector = InteractivePolygon(ax, image, image_path, initial_vertices, on_extract=_on_extract_callback, close_on_extract=True, ignore_enter=True)

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

    # 'Change file' button to return control to startup file selection
    btn_change_ax = plt.axes([0.84, 0.76, 0.12, 0.05])
    btn_change = Button(btn_change_ax, 'Change file')

    change_request = {'path': None}

    def _change_file(event):
        try:
            new_path = choose_image_file(initialfile=os.path.basename(image_path) if image_path else 'Hipp2.1.tiff')
            if not new_path:
                print('Change file canceled.')
                return
            change_request['path'] = new_path
            print(f'Changing file to: {new_path}')
            try:
                import matplotlib.pyplot as _plt
                _plt.close(fig)
            except Exception:
                pass
        except Exception as e:
            print(f'Error selecting new file: {e}')

    btn_change.on_clicked(_change_file)

    # Checkbox to allow skipping cell detection in downstream pipeline
    skip_detection = {'value': False}

    check_ax = plt.axes([0.84, 0.68, 0.12, 0.06])
    check = CheckButtons(check_ax, ['Skip segmentation'], [False])

    def _on_toggle_skip(label):
        try:
            status = check.get_status()[0]
            skip_detection['value'] = bool(status)
        except Exception:
            pass

    check.on_clicked(_on_toggle_skip)

    fig.canvas.draw()
    plt.show()

    if change_request.get('path'):
        return ('change_file', change_request['path'])

    return selector, store, skip_detection.get('value', False)
