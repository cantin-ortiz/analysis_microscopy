"""
Startup helper: image-selection dialog moved out of main.
"""
import os
import tkinter as tk
from tkinter import filedialog


def choose_image_file(initialfile='Hipp2.1.tiff', initialdir=None, filetypes=None, title='Select image file'):
    """Open a Tk file-open dialog and return the selected path or None.

    Args:
        initialfile: suggested filename to show in dialog
        initialdir: starting directory (defaults to cwd)
        filetypes: list of (label, pattern) tuples for file types
        title: dialog title

    Returns:
        Absolute path to the selected file as a string, or None if canceled.
    """
    root = tk.Tk()
    root.withdraw()
    try:
        if initialdir is None:
            initialdir = os.getcwd()
        if filetypes is None:
            filetypes = [
                ('TIFF files', '*.tif *.tiff'),
                ('All images', '*.png *.jpg *.jpeg *.tif *.tiff'),
                ('All files', '*.*')
            ]
        path = filedialog.askopenfilename(title=title, initialdir=initialdir, initialfile=initialfile, filetypes=filetypes)
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    return path or None
