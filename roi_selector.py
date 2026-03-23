"""
ROI selector helper moved out of `main.py`.
"""
import os
import numpy as np
from skimage import io, transform
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import messagebox
import sys
try:
    import czifile
    CZI_AVAILABLE = True
except ImportError:
    CZI_AVAILABLE = False

from interactive import InteractivePolygon
from analysis_store import AnalysisStore
from matplotlib.widgets import Button, CheckButtons, Slider
from startup import choose_image_file


def downsample_if_large(image, max_dimension=2000):
    """Downsample image if either dimension exceeds max_dimension.
    
    Args:
        image: 2D numpy array (grayscale image)
        max_dimension: maximum allowed dimension before downsampling
        
    Returns:
        Downsampled image if needed, otherwise original image
        Prints a message if downsampling occurs
    """
    height, width = image.shape[:2]
    
    if height <= max_dimension and width <= max_dimension:
        return image
    
    # Calculate scale factor to fit within max_dimension
    scale_factor = min(max_dimension / height, max_dimension / width)
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    print(f"\nDownsampling image from {height}x{width} to {new_height}x{new_width} (scale: {scale_factor:.3f})")
    print(f"Original size exceeds {max_dimension}x{max_dimension} threshold")
    
    # Use anti_aliasing to prevent artifacts
    downsampled = transform.resize(image, (new_height, new_width), 
                                   preserve_range=True, 
                                   anti_aliasing=True)
    
    # Preserve original dtype
    downsampled = downsampled.astype(image.dtype)
    
    return downsampled


def load_image(image_path):
    """Load image from .tif/.tiff or .czi file."""
    _, ext = os.path.splitext(image_path.lower())
    
    if ext == '.czi':
        if not CZI_AVAILABLE:
            raise ImportError("czifile library is not installed. Install it with: pip install czifile")
        
        # Read CZI file
        with czifile.CziFile(image_path) as czi:
            image_data = czi.asarray()
            print(f"CZI original shape: {image_data.shape}, dtype: {image_data.dtype}")
            
            # CZI files often have shape like (1, 1, C, Z, Y, X, 1) or similar
            # We need to squeeze out singleton dimensions and select appropriate slice
            image_data = np.squeeze(image_data)
            print(f"After squeeze: {image_data.shape}")
            
            # If still multi-dimensional after squeeze, we need to be smarter about selection
            # Common CZI dimension orders: (C, Z, Y, X) or (Z, C, Y, X) or (C, Y, X) etc.
            if image_data.ndim > 2:
                # Try to find the 2D image plane
                # Last two dimensions are typically Y, X (height, width)
                # If we have more dimensions, try different strategies
                
                if image_data.ndim == 3:
                    # Could be (C, Y, X) or (Z, Y, X)
                    # Take max projection along first axis if it's small (likely channels/z-stack)
                    if image_data.shape[0] <= 10:
                        print(f"Taking max projection along axis 0 (size {image_data.shape[0]})")
                        image_data = np.max(image_data, axis=0)
                    else:
                        # First dimension is likely spatial, take the middle slice
                        mid_idx = image_data.shape[0] // 2
                        print(f"Taking middle slice {mid_idx} from axis 0")
                        image_data = image_data[mid_idx]
                        
                elif image_data.ndim == 4:
                    # Could be (C, Z, Y, X) - take max projection over Z, first channel
                    print(f"4D data: taking max projection")
                    # Take first channel
                    image_data = image_data[0]
                    # Max project over remaining axis if not 2D yet
                    if image_data.ndim > 2:
                        image_data = np.max(image_data, axis=0)
                        
                else:
                    # For higher dimensions, keep reducing
                    while image_data.ndim > 2:
                        image_data = image_data[0]
            
            print(f"Final CZI shape: {image_data.shape}, dtype: {image_data.dtype}")
            print(f"CZI value range: [{image_data.min()}, {image_data.max()}]")
            
            # Convert to float and normalize if needed for display
            # Some CZI files may have unusual data types or ranges
            if image_data.dtype == np.uint16 or image_data.max() > 255:
                # Normalize to 0-255 range for better display
                img_min, img_max = image_data.min(), image_data.max()
                if img_max > img_min:
                    image_data = ((image_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    print(f"Normalized to uint8 range [0, 255]")
            
            return downsample_if_large(image_data)
    else:
        # Load TIFF or other formats with skimage
        image = io.imread(image_path)
        return downsample_if_large(image)


def select_roi_and_show(image_path="Hipp2.1.tiff"):
    """Load image, handle existing polygon prompt, show main ROI selector and return the selector."""
    image = load_image(image_path)

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
    
    # Downsample if the image is too large (after channel/z-stack processing)
    image = downsample_if_large(image)

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
    
    # Store original image for brightness/contrast adjustments
    image_original = image.copy()
    image_min, image_max = float(image.min()), float(image.max())
    
    # Load saved brightness/contrast values from store
    saved_brightness = store.get('brightness', 0)
    saved_contrast = store.get('contrast', 1.0)
    
    # Display image
    im_display = ax.imshow(image, cmap='gray', vmin=image_min, vmax=image_max)
    title = 'Draw polygon ROI - See instructions in top-left corner'
    if initial_vertices:
        title += ' (LOADED EXISTING POLYGON)'
    ax.set_title(title, fontweight='bold')
    ax.axis('off')

    # Adjust layout to make room for sliders on the left and buttons on the right
    plt.subplots_adjust(left=0.15, right=0.82, top=0.95)
    
    # Brightness slider (left side)
    ax_brightness = plt.axes([0.02, 0.25, 0.02, 0.6])
    slider_brightness = Slider(
        ax_brightness, 'Brightness', -100, 100, valinit=saved_brightness, 
        orientation='vertical', valstep=1
    )
    
    # Contrast slider (left side)
    ax_contrast = plt.axes([0.06, 0.25, 0.02, 0.6])
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
            store.set('brightness', brightness)
            store.set('contrast', contrast)
            
            # Apply contrast (multiply) then brightness (add)
            # Contrast adjustment around the midpoint
            img_mid = (image_min + image_max) / 2.0
            adjusted = (image_original - img_mid) * contrast + img_mid + brightness
            
            # Clip to valid range
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
