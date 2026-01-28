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
                "subareas": {},
                "segmented_cells": []
            }
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