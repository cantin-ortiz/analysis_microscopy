"""
Interactive polygon selector module
"""

import numpy as np
from matplotlib.patches import Polygon
from skimage.draw import polygon

import csv
import os
import tkinter as tk
from tkinter import messagebox


class InteractivePolygon:
    """Interactive polygon selector for ROI selection"""

    def __init__(self, ax, image, image_path, initial_vertices=None, on_extract=None, close_on_extract=True, ignore_enter=False, extra_instructions=None):
        self.ax = ax
        self.image = image
        self.image_path = image_path
        self.vertices = initial_vertices if initial_vertices is not None else []
        self.polygon = None
        self.points = None
        self.dragging_point = None
        self.epsilon = 10  # pixels for point selection
        # Fill toggle state (press 'f' to toggle)
        self.fill_enabled = True
        self.fill_alpha = 0.3
        # Colors for face and edge (store separately so edge remains visible)
        self.face_rgb = (1.0, 1.0, 0.0)  # yellow
        self.edgecolor = 'red'

        # Create polygon and points artists
        facecolor = (*self.face_rgb, self.fill_alpha) if self.fill_enabled else 'none'
        self.polygon = Polygon(np.empty((0, 2)), fill=self.fill_enabled, facecolor=facecolor, edgecolor=self.edgecolor, linewidth=2)
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

        # Optional extra instructions to display in the instruction text box
        self.extra_instructions = extra_instructions if extra_instructions is not None else ''
        # Option to ignore Enter key for finalizing selection (use a button instead)
        self.ignore_enter = bool(ignore_enter)

        # Optional callback invoked when user presses Enter to extract ROI
        self.on_extract = on_extract
        # If True, extract_roi will close the figure (original behavior)
        self.close_on_extract = close_on_extract

        # Create instruction text after mode flags are set so update_instructions
        # can safely reference `self.close_on_extract`.
        self.text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        self.update_instructions()

        # If we loaded vertices, update the plot
        if self.vertices:
            self.update_plot()
        
    def update_instructions(self):
        """Update instruction text"""
        instructions = (
            "Left click: Add point (or click on edge to insert)\n"
            "Right click: Remove point\n"
            "Drag: Move point\n"
            f"Points: {len(self.vertices)}\n"
        )
        # Only show Enter instruction when this selector will perform extract on Enter
        if self.close_on_extract and not getattr(self, 'ignore_enter', False):
            instructions = instructions.replace(f"Points: {len(self.vertices)}\n", "Enter: Extract ROI\n" + f"Points: {len(self.vertices)}\n")
        # Always show other shortcuts
        instructions += "F: Toggle fill (on/off)\nC: Clear polygon"
        if len(self.vertices) >= 3:
            area = self.calculate_area()
            instructions += f"\nArea: {area:.1f} pixels²"
        if self.extra_instructions:
            instructions += '\n' + self.extra_instructions
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
        elif event.key in ('f', 'F'):
            # Toggle fill on/off
            self.toggle_fill()


    def toggle_fill(self):
        """Toggle polygon fill visibility/alpha"""
        self.fill_enabled = not self.fill_enabled
        try:
            if self.fill_enabled:
                self.polygon.set_fill(True)
                self.polygon.set_facecolor((*self.face_rgb, self.fill_alpha))
            else:
                # Disable face fill but keep edge visible
                self.polygon.set_fill(False)
                self.polygon.set_facecolor('none')
        except Exception:
            pass
        self.update_instructions()
        self.ax.figure.canvas.draw_idle()
            
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
        print(f"extract_roi called; vertices={len(self.vertices)}")
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
            print("Calling on_extract callback")
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

            try:
                import matplotlib.pyplot as plt
                plt.close(self.ax.figure)
            except Exception:
                pass
