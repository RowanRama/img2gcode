"""img2gcode – convert images to SVG toolpaths and multi-tool GCode."""

import cv2

# OpenCV's thread pool breaks ("Unknown C++ exception") on Windows once SciPy
# has been loaded in the same process. The images here are small enough that
# a single thread costs little.
cv2.setNumThreads(1)

__version__ = "0.2.0"
