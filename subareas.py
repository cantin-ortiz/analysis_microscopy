"""
Subarea selection module
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider, Button
import json
import tkinter as tk
from tkinter import messagebox, simpledialog
import os

from interactive import InteractivePolygon
from startup import choose_image_file
import sys


def launch_subarea_selector(image, image_path, roi_mask=None, roi_vertices=None, store=None):
    """Open a window to select multiple sub-areas using `InteractivePolygon`."""
    fig, ax = plt.subplots(figsize=(10, 8))
    # Make room on the right for buttons so the help text doesn't overlap the image
    plt.subplots_adjust(bottom=0.25, right=0.78, top=0.94)
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    ax.set_title('Draw sub-areas (press Enter to finish each selection)')

    # Instruction/legend text for subarea selector (passed into InteractivePolygon)
    sub_instructions = (
        "H: Toggle pixel overlay: currently: VISIBLE\n"
        "C: Clear polygon"
    )

    vmin = float(np.min(image))
    vmax = float(np.max(image))
    init_thresh = (vmin + vmax) / 2.0
    # Shrink threshold slider width so it doesn't overlap the right margin
    slider_width = 0.63
    ax_thresh = plt.axes([0.15, 0.10, slider_width, 0.03])
    slider_thresh = Slider(ax_thresh, 'Threshold', vmin, vmax, valinit=init_thresh)
    try:
        slider_thresh.on_changed(lambda val: update_dynamic_overlay_from_vertices())
    except Exception:
        pass

    if roi_mask is None:
        roi_mask = np.ones_like(image, dtype=bool)
    else:
        try:
            roi_mask = roi_mask.astype(bool)
            if roi_mask.shape != image.shape:
                print("Warning: provided roi_mask shape does not match image; ignoring roi_mask")
                roi_mask = np.ones_like(image, dtype=bool)
        except Exception:
            roi_mask = np.ones_like(image, dtype=bool)

    ax_save = plt.axes([0.81, 0.80, 0.15, 0.05])
    ax_finish = plt.axes([0.81, 0.72, 0.15, 0.05])
    btn_save = Button(ax_save, 'Save Subareas')
    btn_finish = Button(ax_finish, 'Finish')

    # Change request signal for returning a new file selection to caller
    change_request = {'path': None}

    subareas = []
    overlay_artists = []
    current_im_overlay = None
    overlays_visible = True

    def on_extract_callback(polygon_selector):
        verts = polygon_selector.vertices
        try:
            result = InteractivePolygon.compute_roi_from_vertices(image, verts)
        except Exception as e:
            print(f"Error computing ROI from vertices: {e}")
            return

        thresh = slider_thresh.val
        mask = result['mask'].astype(bool)
        effective_mask = mask & roi_mask
        count = int(np.count_nonzero((image > thresh) & effective_mask))
        area = int(np.count_nonzero(effective_mask))
        mean_int = float(image[effective_mask].mean()) if area > 0 else 0.0
        subareas.append({'vertices': np.array(verts).tolist(), 'count': count, 'area': area, 'mean': mean_int, 'threshold': float(thresh)})

        patch = Polygon(np.array(verts), fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(patch)
        ax.text(verts[0][0], verts[0][1], str(count), color='yellow', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.6))

        try:
            condition = (image > thresh) & mask
            condition = condition & roi_mask
            n_highlight = int(np.sum(condition))
            print(f"Overlay candidate pixels (finalized): {n_highlight}")
            if n_highlight > 0:
                overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=float)
                overlay[..., 0] = condition.astype(float)
                overlay[..., 3] = 0.5 * condition.astype(float)
                im_overlay = ax.imshow(overlay, interpolation='nearest', origin='upper', zorder=2)
                overlay_artists.append(im_overlay)
            else:
                print("No pixels above threshold inside polygon; no overlay added.")
        except Exception as e:
            print(f"Error creating overlay: {e}")

        fig.canvas.draw_idle()
        try:
            update_info_display()
        except Exception:
            pass

    selector = InteractivePolygon(ax, image, image_path, initial_vertices=None,
                                  on_extract=on_extract_callback, close_on_extract=False,
                                  extra_instructions=sub_instructions)

    ax_help_sub = plt.axes([0.81, 0.88, 0.15, 0.05])
    btn_help_sub = Button(ax_help_sub, 'Hide Help')

    def _toggle_help_sub(event):
        try:
            visible = selector.text.get_visible()
            selector.text.set_visible(not visible)
            btn_help_sub.label.set_text('Show Help' if visible else 'Hide Help')
            selector.ax.figure.canvas.draw_idle()
        except Exception:
            pass

    btn_help_sub.on_clicked(_toggle_help_sub)

    def refresh_dynamic_overlay(event):
        nonlocal current_im_overlay
        if event.inaxes != ax:
            return
        verts = selector.vertices
        if not verts or len(verts) < 3:
            if current_im_overlay is not None:
                try:
                    current_im_overlay.set_data(np.zeros((image.shape[0], image.shape[1], 4)))
                except Exception:
                    pass
                fig.canvas.draw_idle()
            return

        try:
            result = InteractivePolygon.compute_roi_from_vertices(image, verts)
            cond = (image > slider_thresh.val) & result['mask'].astype(bool)
            cond = cond & roi_mask
            n = int(np.sum(cond))
            overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=float)
            if n > 0:
                overlay[..., 0] = cond.astype(float)
                overlay[..., 3] = 0.5 * cond.astype(float)

            if current_im_overlay is None:
                current_im_overlay = ax.imshow(overlay, interpolation='nearest', origin='upper', zorder=4)
            else:
                current_im_overlay.set_data(overlay)

        except Exception as e:
            print(f"Error updating dynamic overlay: {e}")

        fig.canvas.draw_idle()
        try:
            update_info_display()
        except Exception:
            pass

    cid_motion_sub = fig.canvas.mpl_connect('motion_notify_event', refresh_dynamic_overlay)

    def update_dynamic_overlay_from_vertices():
        nonlocal current_im_overlay
        verts = selector.vertices
        if not verts or len(verts) < 3:
            if current_im_overlay is not None:
                try:
                    current_im_overlay.set_data(np.zeros((image.shape[0], image.shape[1], 4)))
                except Exception:
                    pass
                fig.canvas.draw_idle()
            return

        try:
            result = InteractivePolygon.compute_roi_from_vertices(image, verts)
            cond = (image > slider_thresh.val) & result['mask'].astype(bool)
            cond = cond & roi_mask
            overlay = np.zeros((image.shape[0], image.shape[1], 4), dtype=float)
            if cond.any():
                overlay[..., 0] = cond.astype(float)
                overlay[..., 3] = 0.5 * cond.astype(float)

            if current_im_overlay is None:
                current_im_overlay = ax.imshow(overlay, interpolation='nearest', origin='upper', zorder=4)
            else:
                current_im_overlay.set_data(overlay)

        except Exception as e:
            print(f"Error updating dynamic overlay: {e}")

        fig.canvas.draw_idle()
        try:
            update_info_display()
        except Exception:
            pass

    def on_button_release(event):
        try:
            if event.inaxes == ax_thresh:
                update_dynamic_overlay_from_vertices()
        except Exception:
            pass

    cid_button_release = fig.canvas.mpl_connect('button_release_event', on_button_release)

    def on_key_sub(event):
        nonlocal overlays_visible, current_im_overlay
        if event.key == 'h':
            overlays_visible = not overlays_visible
            for art in overlay_artists:
                try:
                    art.set_visible(overlays_visible)
                except Exception:
                    pass
            if current_im_overlay is not None:
                try:
                    current_im_overlay.set_visible(overlays_visible)
                except Exception:
                    pass
            fig.canvas.draw_idle()
            state = 'VISIBLE' if overlays_visible else 'HIDDEN'
            print(f"Subarea overlays {state}")
            try:
                selector.extra_instructions = (
                    f"H: Toggle pixel overlay: currently: {state}\n"
                    "C: Clear polygon"
                )
                selector.update_instructions()
            except Exception:
                pass

    cid_key_sub = fig.canvas.mpl_connect('key_press_event', on_key_sub)

    ax_info = plt.axes([0.81, 0.60, 0.15, 0.12])
    ax_info.axis('off')
    info_text = ax_info.text(0.01, 0.98,
                             "Area: 0 px\nSelected: 0 px\n% Selected: 0.0%",
                             transform=ax_info.transAxes, verticalalignment='top', fontsize=10)

    def update_info_display():
        verts = selector.vertices
        if not verts or len(verts) < 3:
            info_text.set_text("Area: 0 px\nSelected: 0 px\n% Selected: 0.0%")
            fig.canvas.draw_idle()
            return
        try:
            result = InteractivePolygon.compute_roi_from_vertices(image, verts)
            mask = result['mask'].astype(bool)
            effective_mask = mask & roi_mask
            area = int(np.count_nonzero(effective_mask))
            thresh = float(slider_thresh.val)
            sel = int(np.count_nonzero((image > thresh) & effective_mask)) if area > 0 else 0
            pct = (sel / area * 100.0) if area > 0 else 0.0
            info_text.set_text(f"Area: {area} px\nSelected: {sel} px\n% Selected: {pct:.1f}%")
        except Exception:
            info_text.set_text("Area: -\nSelected: -\n% Selected: -")
        fig.canvas.draw_idle()

    def save_subareas(event):
        try:
            current_verts = selector.vertices
        except Exception:
            current_verts = None

        if current_verts and len(current_verts) >= 3:
            try:
                result = InteractivePolygon.compute_roi_from_vertices(image, current_verts)
                thresh = slider_thresh.val
                mask = result['mask'].astype(bool)
                effective_mask = mask & roi_mask
                count = int(np.count_nonzero((image > thresh) & effective_mask))
                area = int(np.count_nonzero(effective_mask))
                mean_int = float(image[effective_mask].mean()) if area > 0 else 0.0
                subareas.append({'vertices': np.array(current_verts).tolist(), 'count': count, 'area': area, 'mean': mean_int, 'threshold': float(thresh)})
                try:
                    patch = Polygon(np.array(current_verts), fill=False, edgecolor='blue', linewidth=2)
                    ax.add_patch(patch)
                    ax.text(current_verts[0][0], current_verts[0][1], str(count), color='yellow', fontsize=10,
                            bbox=dict(facecolor='black', alpha=0.6))
                except Exception:
                    pass
            except Exception as e:
                print(f"Error computing stats for current polygon: {e}")

        if len(subareas) == 0:
            print('No subareas to save')
            return
        root = tk.Tk()
        root.withdraw()
        try:
            area_name = simpledialog.askstring('Area name', 'Enter area name:', initialvalue='area1', parent=root)
        except Exception:
            area_name = None

        if area_name is None:
            print('Save cancelled by user')
            root.destroy()
            return

        # Only keep subarea information here; other metadata is stored elsewhere
        payload = []

        for idx, sa in enumerate(subareas, 1):
            verts = sa.get('vertices', [])
            area_px = int(sa.get('area', 0))
            selected = int(sa.get('count', 0))
            pct = (selected / area_px * 100.0) if area_px > 0 else 0.0
            thresh = float(sa.get('threshold', 0.0))
            mean_int = float(sa.get('mean', 0.0))
            payload.append({
                'areaname': area_name,
                'subarea_id': idx,
                'vertices': verts,
                'area_pixels': area_px,
                'selected_pixels': selected,
                'percent_selected': round(pct, 2),
                'threshold': thresh,
                'mean_intensity': mean_int
            })

        try:
            if store is not None:
                # Only save subareas into the AnalysisStore; other keys are managed elsewhere
                try:
                    store.set('subareas', payload)
                    print(f"Subarea results saved to analysis JSON: {store.filename}")
                    try:
                        messagebox.showinfo('Save Complete', f"Subarea results saved to:\n{os.path.basename(store.filename)}", parent=root)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Failed to save subareas to AnalysisStore: {e}")
            else:
                root.destroy()
                raise FileNotFoundError(f"No analysis JSON found at: {json_path}; cannot save subareas")
        except Exception as e:
            print(f"Failed to write JSON file: {e}")
        finally:
            root.destroy()

    def finish(event):
        try:
            root = tk.Tk()
            root.withdraw()
            resp = messagebox.askyesno(
                "Finished",
                "Do you want to continue?.\n\nYes = Select another file\nNo = Close the app",
                parent=root,
                icon='question'
            )
            root.destroy()
        except Exception:
            resp = False

        if resp:
            try:
                new_path = choose_image_file(initialfile=os.path.basename(image_path) if image_path else 'Hipp2.1.tiff')
                if not new_path:
                    print('No new file selected. Exiting.')
                    sys.exit(0)
                change_request['path'] = new_path
                try:
                    import matplotlib.pyplot as _plt
                    _plt.close(fig)
                except Exception:
                    pass
            except Exception as e:
                print(f'Error selecting new file after finish: {e}')
                sys.exit(0)
        else:
            print('Closing application as requested.')
            sys.exit(0)

    btn_save.on_clicked(save_subareas)
    btn_finish.on_clicked(finish)

    plt.show()

    # If finish triggered a change-request, return it to caller
    if change_request.get('path'):
        return ('change_file', change_request['path'])

    return
