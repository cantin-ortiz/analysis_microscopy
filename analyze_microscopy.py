"""
Microscopy image analysis script with interactive polygon ROI selection and cell counting
"""

import numpy as np
from skimage import io
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider, Button
from skimage.draw import polygon
from cellpose import models
import csv
import os
import tkinter as tk
from tkinter import messagebox


# Set SKIP_CELL_DETECTION = True during development to speed up testing
SKIP_CELL_DETECTION = False


class InteractivePolygon:
    """Interactive polygon selector for ROI selection"""
    
    def __init__(self, ax, image, image_path, initial_vertices=None):
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
        
        self.text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.update_instructions()
        
        # If we loaded vertices, update the plot
        if self.vertices:
            self.update_plot()
        
    def update_instructions(self):
        """Update instruction text"""
        instructions = (
            "Left click: Add point\n"
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
                # Add new point
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
        
    def extract_roi(self):
        """Extract pixels within the polygon"""
        if len(self.vertices) < 3:
            print("Need at least 3 points to define a polygon")
            return
            
        # Save polygon coordinates to CSV
        import csv
        import os
        base_name = os.path.splitext(self.image_path)[0]
        csv_path = f"{base_name}_polygon.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])
            for vertex in self.vertices:
                writer.writerow(vertex)
        print(f"\nPolygon coordinates saved to: {csv_path}")
            
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
        
        # Close the polygon selection window
        plt.close(self.ax.figure)
        
        # Launch cell detection interface
        if not SKIP_CELL_DETECTION:
            self.launch_cell_detector(masked_img)
        else:
            print("\nCell detection skipped (SKIP_CELL_DETECTION = True)")
            print("Change SKIP_CELL_DETECTION to False to enable cell counting.")
        
    def launch_cell_detector(self, roi_image):
        """Launch interactive cell detection interface"""
        # Create new figure for cell detection
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.25)
        
        # Initial parameters for Cellpose
        initial_diameter = 30  # Expected cell diameter in pixels
        initial_flow_threshold = 0.4  # Detection confidence (0.1-2.0)
        initial_cellprob_threshold = 0.0  # Cell probability threshold
        initial_min_size = 100  # Filter: min area in pixels²
        
        # Store detection parameters and results
        self.detection_params = {
            'diameter': initial_diameter,
            'flow_threshold': initial_flow_threshold,
            'cellprob_threshold': initial_cellprob_threshold,
            'min_size': initial_min_size
        }
        
        # Display image
        self.det_image = roi_image
        self.det_ax = ax
        self.det_im = ax.imshow(roi_image, cmap='gray')
        self.det_circles = []
        
        # Initialize outline visibility
        self.outlines_visible = True
        
        # Initial detection
        self.detect_cells()
        
        # Create sliders
        ax_diameter = plt.axes([0.15, 0.15, 0.7, 0.03])
        ax_flow = plt.axes([0.15, 0.11, 0.7, 0.03])
        ax_cellprob = plt.axes([0.15, 0.07, 0.7, 0.03])
        ax_min_size = plt.axes([0.15, 0.03, 0.7, 0.03])
        
        self.slider_diameter = Slider(ax_diameter, 'Cell Diameter (px)', 5, 100, 
                                        valinit=initial_diameter, valstep=1)
        self.slider_flow = Slider(ax_flow, 'Flow Threshold', 0.1, 2.0, 
                                        valinit=initial_flow_threshold, valstep=0.05)
        self.slider_cellprob = Slider(ax_cellprob, 'Cell Prob Threshold', -6, 6, 
                                        valinit=initial_cellprob_threshold, valstep=0.5)
        self.slider_min_size = Slider(ax_min_size, 'Min Area (px²)', 10, 1000, 
                                       valinit=initial_min_size, valstep=10)
        
        # Add buttons
        ax_detect = plt.axes([0.60, 0.90, 0.15, 0.05])
        ax_save = plt.axes([0.80, 0.90, 0.15, 0.05])
        self.btn_detect = Button(ax_detect, 'Recompute')
        self.btn_save = Button(ax_save, 'Save Results')
        
        # Connect buttons
        self.btn_detect.on_clicked(self.update_detection)
        self.btn_save.on_clicked(self.save_cell_detection)
        
        # Connect key press event for toggling outlines
        self.cid_key_det = fig.canvas.mpl_connect('key_press_event', self.on_key_det)
        
        plt.show()
        
    def detect_cells(self):
        """Detect cells using Cellpose"""
        # Remove previous circles
        for circle in self.det_circles:
            circle.remove()
        self.det_circles = []
        
        # Initialize Cellpose model
        if not hasattr(self, 'cellpose_model'):
            print("Loading Cellpose model (first run may take a moment)...")
            self.cellpose_model = models.CellposeModel(model_type='cyto2', gpu=True)
            print(f"Cellpose will use GPU: {self.cellpose_model.gpu} (model: cyto2)")
        
        # Run Cellpose segmentation
        print("Running Cellpose segmentation (this may take 30-60 seconds on CPU)...")
        result = self.cellpose_model.eval(
            self.det_image,
            diameter=self.detection_params['diameter'],
            flow_threshold=self.detection_params['flow_threshold'],
            cellprob_threshold=self.detection_params['cellprob_threshold']
        )
        
        # Handle different return formats (version dependent)
        if len(result) == 4:
            masks, flows, styles, diams = result
        else:
            masks = result[0]
        
        # Extract cell properties from masks
        from skimage.measure import regionprops
        regions = regionprops(masks)
        
        # Filter by area
        min_area = self.detection_params['min_size']
        filtered_cells = []
        
        for region in regions:
            if region.area >= min_area:
                y, x = region.centroid
                # Approximate radius from area
                r = np.sqrt(region.area / np.pi)
                filtered_cells.append((y, x, r, region.area))
        
        self.detected_cells = np.array(filtered_cells) if len(filtered_cells) > 0 else np.empty((0, 4))
        
        # Draw circles on detected cells
        for y, x, r, area in filtered_cells:
            circle = Circle((x, y), r, fill=False, 
                           edgecolor='red', linewidth=2, alpha=0.8)
            self.det_ax.add_patch(circle)
            self.det_circles.append(circle)
        
        # Set visibility based on current state
        for circle in self.det_circles:
            circle.set_visible(self.outlines_visible)
        
        # Update title with count and outline status
        status = "VISIBLE" if self.outlines_visible else "HIDDEN"
        self.det_ax.set_title(f'Cell Detection (Cellpose): {len(filtered_cells)} cells detected - Outlines {status}', 
                             fontsize=14, fontweight='bold')
        self.det_ax.axis('off')
        
        print(f"\nDetected {len(filtered_cells)} cells using Cellpose")
        
    def update_detection(self, val):
        """Update cell detection when slider values change"""
        # Update button to show computing state
        self.btn_detect.label.set_text('Computing...')
        self.btn_detect.color = 'orange'
        self.btn_detect.hovercolor = 'orange'
        self.btn_detect.ax.figure.canvas.draw()
        self.btn_detect.ax.figure.canvas.flush_events()
        plt.pause(0.1)  # Force GUI update before computation
        
        # Update parameters
        self.detection_params['diameter'] = self.slider_diameter.val
        self.detection_params['flow_threshold'] = self.slider_flow.val
        self.detection_params['cellprob_threshold'] = self.slider_cellprob.val
        self.detection_params['min_size'] = self.slider_min_size.val
        
        # Run detection
        self.detect_cells()
        
        # Reset button
        self.btn_detect.label.set_text('Recompute')
        self.btn_detect.color = '0.85'
        self.btn_detect.hovercolor = '0.95'
        plt.draw()
        
    def save_cell_detection(self, event):
        """Save cell detection results to CSV"""
        import csv
        import os
        
        if len(self.detected_cells) == 0:
            print("No cells detected to save")
            return
            
        base_name = os.path.splitext(self.image_path)[0]
        csv_path = f"{base_name}_cells.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cell_id', 'x', 'y', 'radius', 'area'])
            for idx, cell in enumerate(self.detected_cells, 1):
                y, x, r, area = cell
                writer.writerow([idx, x, y, r, area])
        
        # Save Cellpose settings to separate CSV
        settings_csv_path = f"{base_name}_settings.csv"
        with open(settings_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['parameter', 'value'])
            for param, value in self.detection_params.items():
                writer.writerow([param, value])
        
        print(f"\nCell detection results saved to: {csv_path}")
        print(f"Cellpose settings saved to: {settings_csv_path}")
        print(f"Total cells: {len(self.detected_cells)}")
        print(f"Parameters used:")
        print(f"  Cell diameter: {self.detection_params['diameter']} px")
        print(f"  Flow threshold: {self.detection_params['flow_threshold']}")
        print(f"  Cell prob threshold: {self.detection_params['cellprob_threshold']}")
        print(f"  Min size: {self.detection_params['min_size']} px²")
        
    def on_key_det(self, event):
        """Handle key press in cell detection interface"""
        if event.key == 'h':  # Toggle hide/show outlines
            self.toggle_outlines()
            
    def toggle_outlines(self):
        """Toggle visibility of cell outlines"""
        self.outlines_visible = not self.outlines_visible
        for circle in self.det_circles:
            circle.set_visible(self.outlines_visible)
        
        # Update title to indicate status
        status = "VISIBLE" if self.outlines_visible else "HIDDEN"
        title = f'Cell Detection (Cellpose): {len(self.detected_cells)} cells detected - Outlines {status}'
        self.det_ax.set_title(title, fontsize=14, fontweight='bold')
        
        self.det_ax.figure.canvas.draw_idle()


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
