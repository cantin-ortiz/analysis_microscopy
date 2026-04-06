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


# Helper wrappers for tkinter messageboxes that create a temporary hidden root
def _mb_showinfo(title, message):
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        try:
            return messagebox.showinfo(title, message, parent=root)
        finally:
            try:
                root.destroy()
            except Exception:
                pass
    except Exception:
        try:
            return messagebox.showinfo(title, message)
        except Exception:
            return None


def _mb_showerror(title, message):
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        try:
            return messagebox.showerror(title, message, parent=root)
        finally:
            try:
                root.destroy()
            except Exception:
                pass
    except Exception:
        try:
            return messagebox.showerror(title, message)
        except Exception:
            return None


def _mb_askyesno(title, message, icon='question'):
    try:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        try:
            return messagebox.askyesno(title, message, parent=root, icon=icon)
        finally:
            try:
                root.destroy()
            except Exception:
                pass
    except Exception:
        try:
            return messagebox.askyesno(title, message, icon=icon)
        except Exception:
            return False


def launch_subarea_selector(image, image_path, roi_mask=None, roi_vertices=None, store=None):
    """Open a window to select multiple sub-areas using `InteractivePolygon`."""
    fig, ax = plt.subplots(figsize=(10, 8))
    # Try to open the figure maximized (preserves window controls on Windows)
    try:
        mgr = plt.get_current_fig_manager()
        try:
            win = mgr.window
            try:
                # Try Windows 'zoomed' first (preserves title bar with min/max/close buttons)
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
    # Make room on the left for sliders and right for buttons
    plt.subplots_adjust(left=0.15, bottom=0.25, right=0.78, top=0.94)
    
    # Store original image for brightness/contrast adjustments
    image_original = image.copy()
    image_min, image_max = float(image.min()), float(image.max())
    
    # Load saved brightness/contrast values from store if available
    saved_brightness = store.get('brightness', 0) if store else 0
    saved_contrast = store.get('contrast', 1.0) if store else 1.0
    
    im_display = ax.imshow(image, cmap='gray', vmin=image_min, vmax=image_max)
    ax.axis('off')
    ax.set_title('Draw sub-areas (press Enter to finish each selection)')
    
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
            
            # Save to store after every adjustment
            if store:
                store.set('brightness', brightness)
                store.set('contrast', contrast)
            
            # Apply contrast (multiply) then brightness (add)
            img_mid = (image_min + image_max) / 2.0
            adjusted = (image_original - img_mid) * contrast + img_mid + brightness
            adjusted = np.clip(adjusted, image_min, image_max)
            
            # Update displayed image
            im_display.set_data(adjusted)
            fig.canvas.draw_idle()
        except Exception as e:
            print(f"Error updating image display: {e}")
    
    slider_brightness.on_changed(update_image_display)
    slider_contrast.on_changed(update_image_display)
    
    # Apply initial brightness/contrast if values were loaded
    if saved_brightness != 0 or saved_contrast != 1.0:
        update_image_display()

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
    saved_artists = []
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

    def clear_saved_artists():
        nonlocal saved_artists, overlay_artists
        try:
            for art in saved_artists:
                try:
                    art.remove()
                except Exception:
                    pass
            saved_artists.clear()
        except Exception:
            pass

        try:
            # remove overlays tracked separately and clear the list
            for art in list(overlay_artists):
                try:
                    art.remove()
                except Exception:
                    pass
            overlay_artists.clear()
        except Exception:
            pass

        try:
            fig.canvas.draw_idle()
        except Exception:
            pass

    def render_saved_subareas():
        nonlocal saved_artists, overlay_artists
        if store is None:
            return
        try:
            existing = store.get('subareas', {}) or {}
            if not isinstance(existing, dict):
                existing = {}

            for area_name, sa in existing.items():
                try:
                    verts = sa.get('vertices', [])
                    if not verts or len(verts) < 3:
                        continue

                    try:
                        patch = Polygon(np.array(verts), fill=False, edgecolor='green', linewidth=2)
                        ax.add_patch(patch)
                        saved_artists.append(patch)
                    except Exception:
                        pass

                    try:
                        label = area_name
                        txt = ax.text(verts[0][0], verts[0][1], label, color='yellow', fontsize=10,
                                      bbox=dict(facecolor='black', alpha=0.6))
                        saved_artists.append(txt)
                    except Exception:
                        pass

                    pass

                except Exception as e:
                    print(f"Error displaying saved subarea '{area_name}': {e}")

            try:
                fig.canvas.draw_idle()
            except Exception:
                pass
        except Exception as e:
            print(f"Failed to load saved subareas from AnalysisStore: {e}")

    # render any saved subareas at startup
    try:
        render_saved_subareas()
    except Exception:
        pass

    def delete_areas(event):
        # Allow user to select one or more area names to delete from the store
        if store is None:
            _mb_showerror('No analysis JSON', 'No analysis JSON available to delete areas from.')
            return

        try:
            existing = store.get('subareas', {}) or {}
            if not isinstance(existing, dict) or len(existing) == 0:
                _mb_showinfo('No saved areas', 'There are no saved areas to delete.')
                return

            names = list(existing.keys())

            # Build a modal dialog using the matplotlib Tk parent when possible
            parent = None
            try:
                parent = fig.canvas.manager.window
            except Exception:
                try:
                    parent = fig.canvas.get_tk_widget().winfo_toplevel()
                except Exception:
                    parent = None

            dlg_parent = parent
            if dlg_parent is None:
                # fallback to creating a temporary root
                dlg_parent = tk.Tk()
                dlg_parent.withdraw()

            dlg = tk.Toplevel(dlg_parent)
            dlg.title('Delete Areas')
            try:
                lbl = tk.Label(dlg, text='Select areas to delete:')
                lbl.pack(padx=10, pady=(10, 0))
                listbox = tk.Listbox(dlg, selectmode=tk.MULTIPLE, width=40, height=min(12, max(4, len(names))))
                for n in names:
                    listbox.insert(tk.END, n)
                listbox.pack(padx=10, pady=10)

                btn_frame = tk.Frame(dlg)
                btn_frame.pack(padx=10, pady=(0, 10))

                result = {'selected': None}

                def on_delete_confirm():
                    sel = [listbox.get(i) for i in listbox.curselection()]
                    result['selected'] = sel
                    dlg.destroy()

                def on_cancel():
                    dlg.destroy()

                bdel = tk.Button(btn_frame, text='Delete', command=on_delete_confirm)
                bdel.pack(side='left', padx=(0, 5))
                bcan = tk.Button(btn_frame, text='Cancel', command=on_cancel)
                bcan.pack(side='left')

                dlg.transient(dlg_parent)
                try:
                    dlg.grab_set()
                except Exception:
                    pass
                try:
                    dlg.focus_force()
                except Exception:
                    pass
                dlg.wait_window()
            finally:
                try:
                    if parent is None:
                        dlg_parent.destroy()
                except Exception:
                    pass

            sel = result.get('selected')
            if not sel:
                return

            # Confirm deletion
            ok = _mb_askyesno('Confirm Delete', f"Delete {len(sel)} area(s)?", icon='warning')
            if not ok:
                return

            # Delete selected keys and write back to store
            try:
                for name in sel:
                    if name in existing:
                        existing.pop(name, None)
                # If no remaining entries, set to empty dict
                store.set('subareas', existing)
            except Exception as e:
                _mb_showerror('Delete Failed', f'Failed to delete areas: {e}')
                return

            # Refresh displayed saved subareas
            try:
                clear_saved_artists()
                render_saved_subareas()
            except Exception as e:
                print(f"Error refreshing display after delete: {e}")

            _mb_showinfo('Deleted', f'Deleted {len(sel)} area(s).')
        except Exception as e:
            print(f"Error in delete_areas: {e}")


    ax_help_sub = plt.axes([0.81, 0.88, 0.15, 0.05])
    btn_help_sub = Button(ax_help_sub, 'Hide Help')

    ax_delete = plt.axes([0.81, 0.64, 0.15, 0.05])
    btn_delete = Button(ax_delete, 'Delete Area')
    btn_delete.on_clicked(delete_areas)

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

    # moved down to avoid overlapping the right-side buttons
    # Compute um² per pixel for physical-area conversion in subareas
    sub_um2_per_pixel = None
    if store is not None:
        ps = store.get('pixel_size_um')
        if ps is not None:
            try:
                sub_um2_per_pixel = ps['x'] * ps['y']
            except Exception:
                pass

    ax_info = plt.axes([0.81, 0.46, 0.15, 0.12])
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
            area_str = f"Area: {area} px"
            if sub_um2_per_pixel is not None:
                area_mm2 = area * sub_um2_per_pixel / 1e6
                area_str += f" ({area_mm2:.4f} mm²)"
            info_text.set_text(f"{area_str}\nSelected: {sel} px\n% Selected: {pct:.1f}%")
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
            except Exception as e:
                print(f"Error computing stats for current polygon: {e}")

        if len(subareas) == 0:
            print('No subareas to save')
            return
        # Prefer using the matplotlib Tk parent if available to avoid creating extra Tk roots
        parent = None
        try:
            parent = fig.canvas.manager.window
        except Exception:
            try:
                parent = fig.canvas.get_tk_widget().winfo_toplevel()
            except Exception:
                parent = None

        try:
            if parent is not None:
                area_name = simpledialog.askstring('Area name', 'Enter area name:', initialvalue='area1', parent=parent)
            else:
                # Fallback to modal Toplevel when no parent available
                root = tk.Tk()
                root.withdraw()
                dlg = tk.Toplevel(root)
                dlg.title('Area name')
                lbl = tk.Label(dlg, text='Enter area name:')
                lbl.pack(padx=10, pady=(10, 0))
                entry = tk.Entry(dlg)
                entry.insert(0, 'area1')
                entry.pack(padx=10, pady=10)
                result = {'value': None}

                def on_ok():
                    v = entry.get().strip()
                    result['value'] = v if v else None
                    dlg.destroy()

                def on_cancel():
                    dlg.destroy()

                btn_frame = tk.Frame(dlg)
                btn_frame.pack(padx=10, pady=(0, 10))
                bok = tk.Button(btn_frame, text='OK', command=on_ok)
                bok.pack(side='left', padx=(0, 5))
                bcan = tk.Button(btn_frame, text='Cancel', command=on_cancel)
                bcan.pack(side='left')
                dlg.transient(root)
                try:
                    dlg.grab_set()
                except Exception:
                    pass
                try:
                    dlg.focus_force()
                except Exception:
                    pass
                dlg.wait_window()
                area_name = result.get('value')
                try:
                    dlg.destroy()
                except Exception:
                    pass
                try:
                    root.destroy()
                except Exception:
                    pass
        except Exception:
            area_name = None

        if area_name is None:
            print('Save cancelled by user')
            return
        # Only keep subarea information here; other metadata is stored elsewhere
        # Store a single subarea per area name — use the last drawn subarea if multiple
        if len(subareas) == 0:
            print('No subareas to save')
            try:
                root.destroy()
            except Exception:
                pass
            return

        sa = subareas[-1]
        verts = sa.get('vertices', [])
        area_px = int(sa.get('area', 0))
        selected = int(sa.get('count', 0))
        pct = (selected / area_px * 100.0) if area_px > 0 else 0.0
        thresh = float(sa.get('threshold', 0.0))
        mean_int = float(sa.get('mean', 0.0))
        payload = {
            'vertices': verts,
            'area_pixels': area_px,
            'selected_pixels': selected,
            'percent_selected': round(pct, 2),
            'threshold': thresh,
            'mean_intensity': mean_int,
        }
        # Add area in mm² if pixel size is available
        if sub_um2_per_pixel is not None:
            payload['area_mm2'] = area_px * sub_um2_per_pixel / 1e6
        else:
            payload['area_mm2'] = None

        try:
            if store is not None:
                # Save subareas into the AnalysisStore as a mapping: areaname -> list of subarea objects
                try:
                    existing = store.get('subareas', {}) or {}
                    # ensure we're working with a dict
                    if not isinstance(existing, dict):
                        existing = {}

                    # If an entry with this area name exists, ask before overwriting
                    if area_name in existing:
                        try:
                            overwrite = _mb_askyesno('Overwrite Subareas', f"Subareas named '{area_name}' already exist in the analysis file. Overwrite them?", icon='warning')
                        except Exception:
                            overwrite = False

                        if not overwrite:
                            print('Save cancelled by user (did not overwrite existing subareas).')
                            return

                    # Replace (or add) the entry for this area name with the new single subarea object
                    existing[area_name] = payload
                    store.set('subareas', existing)

                    # Refresh displayed saved subareas so updates replace previous drawings
                    try:
                        clear_saved_artists()
                        render_saved_subareas()
                    except Exception as e:
                        print(f"Error refreshing displayed subareas after save: {e}")

                    # Clear the interactive polygon so user can start a new selection
                    try:
                        selector.vertices = []
                        selector.update_plot()
                    except Exception:
                        pass

                    print(f"Subarea results saved to analysis JSON: {store.filename}")
                    try:
                        _mb_showinfo('Save Complete', f"Subarea results saved to:\n{os.path.basename(store.filename)}")
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Failed to save subareas to AnalysisStore: {e}")
            else:
                try:
                    root.destroy()
                except Exception:
                    pass
                raise FileNotFoundError(f"No analysis JSON found for image: {image_path}; cannot save subareas")
        except Exception as e:
            print(f"Failed to write JSON file: {e}")
        finally:
            try:
                root.destroy()
            except Exception:
                pass

    def finish(event):
        try:
            resp = _mb_askyesno("Finished", "Do you want to continue?.\n\nYes = Select another file\nNo = Close the app", icon='question')
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
