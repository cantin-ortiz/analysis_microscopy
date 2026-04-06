import json
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional


class AnalysisStore:
    """Simple JSON-backed store for analysis data.

    Stores data in a JSON file (default: "filename_analysis.json") with a
    predictable structure:

    {
      "polygons": { name: {"points": [...], "metadata": {...}}, ... },
      "metadata": { ... }
    }

    Writes are atomic (via temporary file + os.replace). Methods save to disk
    automatically after mutating operations.
    """

    def __init__(self, input_file: str = "image.tif") -> None:
        # keep the provided path so we can create the analysis file next to it
        self.input_file = input_file
        base = os.path.splitext(os.path.basename(self.input_file))[0]
        parent = os.path.dirname(self.input_file) or os.getcwd()
        self.filename = os.path.join(parent, f"{base}_analysis.json")
        self._load_or_create()

    def _load_or_create(self) -> None:
        if os.path.exists(self.filename):
            try:
                self.load()
                self._migrate()
            except Exception as exc:
                # On load failure, do not silently overwrite existing file.
                raise RuntimeError(f"Failed to load analysis file '{self.filename}': {exc}") from exc
        else:
            # ensure parent dir exists
            parent = os.path.dirname(self.filename)
            if parent and not os.path.exists(parent):
                os.makedirs(parent, exist_ok=True)
            # create a sensible default payload including creation time
            self.data = {
                "input_file": self.input_file,
                "filename": self.filename,
                "creation_time": datetime.now().isoformat(),
                "roi_polygon": [],
                "pixel_size_um": None,
                "pixel_size_um_manual": False,
                "roi_area_pixels": None,
                "roi_area_mm2": None,
                "subareas": {},
                "segmented_cells": []
            }
            self.save()

    @staticmethod
    def _shoelace_area(polygon):
        """Compute area of a polygon using the Shoelace formula (pure Python)."""
        n = len(polygon)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
        return abs(area) / 2.0

    def _migrate(self) -> None:
        """Add missing keys for backward compatibility with older analysis files."""
        changed = False
        new_keys = {
            'pixel_size_um': None,
            'pixel_size_um_manual': False,
            'roi_area_pixels': None,
            'roi_area_mm2': None,
        }
        for key, default in new_keys.items():
            if key not in self.data:
                self.data[key] = default
                changed = True

        # If we have a polygon but no pixel area, compute from stored vertices
        if self.data.get('roi_area_pixels') is None and self.data.get('roi_polygon'):
            polygon = self.data['roi_polygon']
            if len(polygon) >= 3:
                try:
                    self.data['roi_area_pixels'] = self._shoelace_area(polygon)
                    changed = True
                except Exception:
                    pass

        # Backfill area_mm2 for subareas that only have area_pixels
        pixel_size = self.data.get('pixel_size_um')
        if pixel_size is not None:
            try:
                um2 = pixel_size['x'] * pixel_size['y']
                subareas = self.data.get('subareas', {})
                if isinstance(subareas, dict):
                    for name, sa in subareas.items():
                        if isinstance(sa, dict) and 'area_pixels' in sa and 'area_mm2' not in sa:
                            sa['area_mm2'] = sa['area_pixels'] * um2 / 1e6
                            changed = True
            except Exception:
                pass

        if changed:
            self.save()

    def load(self) -> None:
        with open(self.filename, "r", encoding="utf-8") as fh:
            self.data = json.load(fh)

    def reload(self) -> None:
        """Reload data from disk."""
        self.load()

    def save(self) -> None:
        """Atomically write current data to disk."""
        self._atomic_write(self.data)

    def _atomic_write(self, data: Dict[str, Any]) -> None:
        directory = os.path.dirname(self.filename) or "."
        fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_analysis_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.filename)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    # Generic accessors
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self.save()


if __name__ == '__main__':
    AnalysisStore("Hipp2.1.tiff")