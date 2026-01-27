"""
Microscopy image analysis script with interactive polygon ROI selection and cell counting
"""

import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button, PolygonSelector
from skimage.draw import polygon
from cellpose import models
import csv
import os
import tkinter as tk
from tkinter import messagebox


# Set SKIP_CELL_DETECTION = True during development to speed up testing
SKIP_CELL_DETECTION = True


class InteractivePolygon:
    """Interactive polygon selector for ROI selection"""
    
    def __init__(self, ax, image, image_path, initial_vertices=None, on_extract=None, close_on_extract=True):
        self.ax = ax
        self.image = image
        self.image_path = image_path
        self.vertices = initial_vertices if initial_vertices is not None else []
        self.polygon = None
        self.points = None
        self.dragging_point = None
        self.epsilon = 10  # pixels for point selection
        
        # Create polygon and points artists
        self.polygon = Polygon(np.empty((0, 2)), fill=True, alpha=0.3, facecolor='yellow', edgecolor='red', linewidth=2)
        self.ax.add_patch(self.polygon)
        self.points, = self.ax.plot([], [], 'ro', markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        
        # Connect events
        self.cid_press = self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.cid_key = self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key)

        # Disable other key_press_event callbacks on this canvas to avoid
        # unintended toolbar or global handlers reacting to keypresses
        try:
            canvas = self.ax.figure.canvas
            callbacks = canvas.callbacks.callbacks.get('key_press_event', {})
            # Disconnect any callback that is not the one we just added.
            for cid, func in list(callbacks.items()):
                if cid != self.cid_key:
                    try:
                        canvas.mpl_disconnect(cid)
                    except Exception:
                        pass
        except Exception:
            pass
        
        self.text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.update_instructions()
        
        # Optional callback invoked when user presses Enter to extract ROI
        self.on_extract = on_extract
        # If True, extract_roi will close the figure (original behavior)
        self.close_on_extract = close_on_extract

        # If we loaded vertices, update the plot
        if self.vertices:
            self.update_plot()
        
    def update_instructions(self):
        """Update instruction text"""
        instructions = (
            "Left click: Add point (or click on edge to insert)\n"
            "Right click: Remove nearest point\n"
            "Drag: Move point\n"
            "Enter: Extract ROI\n"
            "C: Clear polygon\n"
            f"Points: {len(self.vertices)}"
        )
        if len(self.vertices) >= 3:
            area = self.calculate_area()
            instructions += f"\nArea: {area:.1f} pixels²"
        self.text.set_text(instructions)
        
    def calculate_area(self):
        """Calculate polygon area using Shoelace formula"""
        if len(self.vertices) < 3:
            return 0
        vertices = np.array(self.vertices)
        x = vertices[:, 0]
        y = vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        
    def get_nearest_vertex_idx(self, x, y):
        """Find nearest vertex to (x, y)"""
        if not self.vertices:
            return None
        distances = [np.hypot(vx - x, vy - y) for vx, vy in self.vertices]
        min_idx = np.argmin(distances)
        if distances[min_idx] < self.epsilon:
            return min_idx
        return None
        
    def get_edge_insertion_point(self, x, y):
        """Check if point (x, y) is on a polygon edge and return insertion index"""
        if len(self.vertices) < 2:
            return None
            
        for i in range(len(self.vertices)):
            # Get current edge (from vertex i to i+1, wrapping around)
            p1 = np.array(self.vertices[i])
            p2 = np.array(self.vertices[(i + 1) % len(self.vertices)])
            
            # Calculate distance from point to line segment
            dist = self.point_to_line_distance(p1, p2, np.array([x, y]))
            
            if dist < self.epsilon:
                # Point is close to this edge, return insertion index
                # Insert after vertex i
                return (i + 1) % len(self.vertices)
        
        return None
        
    def point_to_line_distance(self, p1, p2, p):
        """Calculate distance from point p to line segment p1-p2"""
        # Vector from p1 to p2
        line_vec = p2 - p1
        # Vector from p1 to p
        point_vec = p - p1
        
        # Length of line segment
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec)
        
        # Project point onto line
        proj = np.dot(point_vec, line_vec) / (line_len ** 2)
        proj = np.clip(proj, 0, 1)  # Clamp to segment
        
        # Closest point on segment
        closest = p1 + proj * line_vec
        
        # Distance to closest point
        return np.linalg.norm(p - closest)
        
    def on_press(self, event):
        """Handle mouse press"""
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:  # Left click
            # Check if clicking near existing point
            idx = self.get_nearest_vertex_idx(event.xdata, event.ydata)
            if idx is not None:
                self.dragging_point = idx
            else:
                # Check if clicking on an edge
                insert_idx = self.get_edge_insertion_point(event.xdata, event.ydata)
                if insert_idx is not None:
                    # Insert new point on edge
                    self.vertices.insert(insert_idx, [event.xdata, event.ydata])
                    self.update_plot()
                else:
                    # Add new point at the end
                    self.vertices.append([event.xdata, event.ydata])
                    self.update_plot()
                
        elif event.button == 3:  # Right click - remove point
            idx = self.get_nearest_vertex_idx(event.xdata, event.ydata)
            if idx is not None:
                self.vertices.pop(idx)
                self.update_plot()
                
    def on_release(self, event):
        """Handle mouse release"""
        self.dragging_point = None
        
    def on_motion(self, event):
        """Handle mouse motion"""
        if self.dragging_point is None:
            return
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        self.vertices[self.dragging_point] = [event.xdata, event.ydata]
        self.update_plot()
        
    def on_key(self, event):
        """Handle key press"""
        if event.key == 'c':  # Clear polygon
            self.vertices = []
            self.update_plot()
        elif event.key == 'enter' and len(self.vertices) >= 3:  # Extract ROI
            self.extract_roi()
            
    def update_plot(self):
        """Update polygon and points visualization"""
        if self.vertices:
            vertices_array = np.array(self.vertices)
            self.points.set_data(vertices_array[:, 0], vertices_array[:, 1])
            if len(self.vertices) >= 3:
                self.polygon.set_xy(vertices_array)
            else:
                self.polygon.set_xy(np.empty((0, 2)))
        else:
            self.points.set_data([], [])
            self.polygon.set_xy(np.empty((0, 2)))
        
        self.update_instructions()
        self.ax.figure.canvas.draw_idle()
        self.ax.figure.canvas.flush_events()  # Process events immediately

    @staticmethod
    def compute_roi_from_vertices(image, vertices):
        """Compute mask, cropped image and ROI statistics from polygon vertices.

        Returns a dict with keys: mask, roi_pixels, masked_img, area, bbox (rmin,rmax,cmin,cmax)
        """
        vertices = np.array(vertices)
        rr, cc = polygon(vertices[:, 1], vertices[:, 0], image.shape)
        mask = np.zeros(image.shape, dtype=bool)
        mask[rr, cc] = True

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return {'mask': mask, 'roi_pixels': np.array([]), 'masked_img': np.zeros((0, 0)), 'area': 0, 'bbox': (0,0,0,0)}
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        cropped_image = image[rmin:rmax+1, cmin:cmax+1]
        cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]
        masked_img = cropped_image.copy()
        masked_img[~cropped_mask] = 0

        roi_pixels = image[mask]
        area = int(np.sum(mask))

        return {
            'mask': mask,
            'roi_pixels': roi_pixels,
            'masked_img': masked_img,
            'area': area,
            'bbox': (rmin, rmax, cmin, cmax)
        }
        
    def extract_roi(self):
        """Extract pixels within the polygon"""
        if len(self.vertices) < 3:
            print("Need at least 3 points to define a polygon")
            return
        # Create mask
        vertices = np.array(self.vertices)
        rr, cc = polygon(vertices[:, 1], vertices[:, 0], self.image.shape)
        
        # Create mask
        mask = np.zeros(self.image.shape, dtype=bool)
        mask[rr, cc] = True
        
        # Create mask
        mask = np.zeros(self.image.shape, dtype=bool)
        mask[rr, cc] = True
        
        # Get bounding box of ROI for cropping
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # Crop to ROI bounding box
        cropped_image = self.image[rmin:rmax+1, cmin:cmax+1]
        cropped_mask = mask[rmin:rmax+1, cmin:cmax+1]
        
        # Apply mask (set pixels outside polygon to zero)
        masked_img = cropped_image.copy()
        masked_img[~cropped_mask] = 0
        
        # Extract ROI statistics
        roi_pixels = self.image[mask]
        print("\n" + "="*50)
        print("ROI Statistics:")
        print("="*50)
        print(f"Number of pixels: {len(roi_pixels)}")
        print(f"Mean intensity: {roi_pixels.mean():.2f}")
        print(f"Std intensity: {roi_pixels.std():.2f}")
        print(f"Min intensity: {roi_pixels.min()}")
        print(f"Max intensity: {roi_pixels.max()}")
        print(f"Polygon area: {self.calculate_area():.1f} pixels²")
        print(f"Cropped image size: {masked_img.shape}")
        
        # Store for later use
        self.mask = mask
        self.roi_pixels = roi_pixels
        self.crop_offset = (rmin, cmin)  # Store offset for coordinate mapping
        # Store additional attributes
        self.masked_img = masked_img
        self.area = int(np.sum(mask))

        # If an on_extract callback is provided, call it instead of closing
        if callable(self.on_extract):
            try:
                # Let the callback handle storing or visualizing the ROI
                self.on_extract(self)
            except Exception as e:
                print(f"on_extract callback raised: {e}")

            # Reset polygon for next selection if we're not closing
            if not self.close_on_extract:
                self.vertices = []
                self.update_plot()
                return

        # Save polygon coordinates only when closing (original behavior)
        if self.close_on_extract:
            import csv
            base_name = os.path.splitext(self.image_path)[0]
            csv_path = f"{base_name}_polygon.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['x', 'y'])
                for vertex in self.vertices:
                    writer.writerow(vertex)
            print(f"\nPolygon coordinates saved to: {csv_path}")

            plt.close(self.ax.figure)
        


# Module-level CELLPOSE model cache
CELLPOSE_MODEL = None


def launch_cell_detector(roi_image, image_path):
    """Launch interactive cell detection interface (module-level)"""
    global CELLPOSE_MODEL

    # Create new figure for cell detection
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(bottom=0.25)

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

        # Initialize Cellpose model
        global CELLPOSE_MODEL
        if CELLPOSE_MODEL is None:
            print("Loading Cellpose model (first run may take a moment)...")
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

    # Create sliders
    ax_diameter = plt.axes([0.15, 0.15, 0.7, 0.03])
    ax_flow = plt.axes([0.15, 0.11, 0.7, 0.03])
    ax_cellprob = plt.axes([0.15, 0.07, 0.7, 0.03])
    ax_min_size = plt.axes([0.15, 0.03, 0.7, 0.03])

    slider_diameter = Slider(ax_diameter, 'Cell Diameter (px)', 5, 100, valinit=initial_diameter, valstep=1)
    slider_flow = Slider(ax_flow, 'Flow Threshold', 0.1, 2.0, valinit=initial_flow_threshold, valstep=0.05)
    slider_cellprob = Slider(ax_cellprob, 'Cell Prob Threshold', -6, 6, valinit=initial_cellprob_threshold, valstep=0.5)
    slider_min_size = Slider(ax_min_size, 'Min Area (px²)', 10, 1000, valinit=initial_min_size, valstep=10)

    ax_detect = plt.axes([0.60, 0.90, 0.15, 0.05])
    ax_save = plt.axes([0.80, 0.90, 0.15, 0.05])
    btn_detect = Button(ax_detect, 'Recompute')
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

    def on_key_det(event):
        nonlocal outlines_visible
        if event.key == 'h':
            outlines_visible = not outlines_visible
            for circle in det_circles:
                circle.set_visible(outlines_visible)
            status = "VISIBLE" if outlines_visible else "HIDDEN"
            det_ax.set_title(f'Cell Detection (Cellpose): {len(detected_cells)} cells detected - Outlines {status}', fontsize=14, fontweight='bold')
            det_ax.figure.canvas.draw_idle()

    btn_detect.on_clicked(update_detection)
    btn_save.on_clicked(save_cell_detection)
    cid_key_det = fig.canvas.mpl_connect('key_press_event', on_key_det)

    # Initial detection
    detect_cells()

    plt.show()

    # After detection window closes, control returns to main flow


def launch_subarea_selector(image, image_path):
    """Open a window to select multiple sub-areas using `InteractivePolygon`.

    This creates a single figure with controls (threshold + Save/Finish) and
    embeds an `InteractivePolygon` on the image axes. Each time the user
    finishes a polygon (presses Enter), the polygon's `on_extract` callback
    will be invoked and the polygon will be reset so the user can draw the
    next sub-area. This ensures identical behavior/appearance to the main
    ROI selector.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    ax.set_title('Draw sub-areas (press Enter to finish each selection)')

    vmin = float(np.min(image))
    vmax = float(np.max(image))
    init_thresh = (vmin + vmax) / 2.0
    ax_thresh = plt.axes([0.15, 0.10, 0.7, 0.03])
    slider_thresh = Slider(ax_thresh, 'Threshold', vmin, vmax, valinit=init_thresh)

    ax_save = plt.axes([0.80, 0.90, 0.15, 0.05])
    ax_finish = plt.axes([0.60, 0.90, 0.15, 0.05])
    btn_save = Button(ax_save, 'Save Subareas')
    btn_finish = Button(ax_finish, 'Finish')

    subareas = []

    def on_extract_callback(polygon_selector):
        # polygon_selector is the InteractivePolygon instance
        verts = polygon_selector.vertices
        try:
            result = InteractivePolygon.compute_roi_from_vertices(image, verts)
        except Exception as e:
            print(f"Error computing ROI from vertices: {e}")
            return

        thresh = slider_thresh.val
        mask = result['mask']
        count = int(np.sum(image[mask] > thresh))
        area = result['area']
        mean_int = float(result['roi_pixels'].mean()) if result['area'] > 0 else 0.0
        subareas.append({'vertices': np.array(verts).tolist(), 'count': count, 'area': area, 'mean': mean_int, 'threshold': float(thresh)})

        # Draw polygon and label
        patch = Polygon(np.array(verts), fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(patch)
        ax.text(verts[0][0], verts[0][1], str(count), color='yellow', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.6))
        fig.canvas.draw_idle()

    # Create an InteractivePolygon on the image axes that does NOT close the
    # figure when the user finishes a polygon; instead it calls
    # `on_extract_callback` and resets for another selection.
    selector = InteractivePolygon(ax, image, image_path, initial_vertices=None,
                                  on_extract=on_extract_callback, close_on_extract=False)

    def save_subareas(event):
        if len(subareas) == 0:
            print('No subareas to save')
            return
        base_name = os.path.splitext(image_path)[0]
        csv_path = f"{base_name}_subareas.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['subarea_id', 'count_above_threshold', 'area_pixels', 'mean_intensity', 'threshold', 'vertices'])
            for idx, sa in enumerate(subareas, 1):
                writer.writerow([idx, sa['count'], sa['area'], sa['mean'], sa['threshold'], sa['vertices']])
        print(f"Subarea results saved to: {csv_path}")
        messagebox.showinfo('Save Complete', f"Subarea results saved to:\n{os.path.basename(csv_path)}")

    def finish(event):
        plt.close(fig)

    btn_save.on_clicked(save_subareas)
    btn_finish.on_clicked(finish)



    plt.show()


# Load the TIFF image
image_path = "Hipp2.1.tiff"
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
    # Create a hidden root window for the dialog
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    # Show dialog box
    response = messagebox.askyesno(
        "Existing Polygon Found",
        f"Found existing polygon file:\n{os.path.basename(polygon_csv_path)}\n\n"
        "Do you want to load it?\n\n"
        "Yes = Load existing polygon\n"
        "No = Start from scratch",
        icon='question'
    )
    
    root.destroy()  # Clean up
    
    if response:  # User clicked Yes
        try:
            with open(polygon_csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                initial_vertices = [[float(row[0]), float(row[1])] for row in reader]
            print(f"Loaded {len(initial_vertices)} polygon points from file")
        except Exception as e:
            print(f"Error loading polygon file: {e}")
            print("Starting with empty polygon")
            initial_vertices = None
    else:  # User clicked No
        print("Starting with empty polygon (existing file will be overwritten when you save)")

# Create interactive plot
fig, ax = plt.subplots(figsize=(12, 10))
ax.imshow(image, cmap='gray')
title = 'Draw polygon ROI - See instructions in top-left corner'
if initial_vertices:
    title += ' (LOADED EXISTING POLYGON)'
ax.set_title(title, fontweight='bold')
ax.axis('off')

# Create interactive polygon selector
selector = InteractivePolygon(ax, image, image_path, initial_vertices)

plt.tight_layout()
plt.show()

# After polygon selection window closes, continue sequentially
if hasattr(selector, 'masked_img') and getattr(selector, 'masked_img') is not None:
    if not SKIP_CELL_DETECTION:
        launch_cell_detector(selector.masked_img, image_path)
    else:
        print("\nCell detection skipped (SKIP_CELL_DETECTION = True)")
        print("Opening subarea selector for pixel-threshold counts.")

    # Always open subarea selector after optional detection
    try:
        launch_subarea_selector(selector.masked_img, image_path)
    except Exception as e:
        print(f"Failed to launch subarea selector: {e}")
else:
    print("No ROI selected. Exiting.")
