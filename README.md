Microscopy analysis — refactored

Quick start
- Install dependencies: `pip install -r requirements.txt`
- Run the GUI: `python main.py`

Project layout
- `main.py`: lightweight entry point that orchestrates the workflow
- `roi_selector.py`: interactive ROI loader + polygon selection (moved out of `main.py`)
- `interactive.py`: `InteractivePolygon` class and ROI utilities
- `detection.py`: interactive StarDist-based cell detection view
- `subareas.py`: subarea selection, threshold overlays, and JSON export

Notes
- `SKIP_CELL_DETECTION` in `main.py` is set to `True` for fast testing; set to `False` to enable StarDist segmentation (requires `stardist` and `tensorflow`).
- Saved outputs: `*_cells.csv` + `*_settings.csv` (cell detection), `*_subareas.json` (subarea stats), and per-image analysis JSON files containing `roi_polygon`.

Testing and development
- To quickly smoke-test imports without opening GUIs run:

```bash
python -c "import main, interactive, detection, subareas, roi_selector; print('ok')"
```

Contributing
- Keep UI code in the small modules above; prefer unit-testing pure functions in `interactive.py` (e.g., `compute_roi_from_vertices`).

If you want, I can add a short test suite and a `black`/`flake8` configuration next.
