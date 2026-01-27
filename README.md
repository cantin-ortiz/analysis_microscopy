Microscopy analysis — refactored

- Run the GUI: `python main.py`
- Requirements: see `requirements.txt` (install with `pip install -r requirements.txt`)

Files:
- `main.py`: entry point
- `interactive.py`: polygon selector
- `detection.py`: cell detection (Cellpose)
- `subareas.py`: subarea selection and JSON export

Notes:
- `SKIP_CELL_DETECTION` in `main.py` can be toggled for faster testing.
