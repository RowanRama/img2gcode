from setuptools import setup, find_packages

setup(
    name="img2gcode",
    version="0.1.0",
    description="Convert raster images to multi-tool GCode toolpaths",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "Pillow>=10.0",
        "scikit-learn>=1.3",
        "opencv-python-headless>=4.8",
        "matplotlib>=3.7",
        "svgelements>=1.9",
    ],
    entry_points={
        "console_scripts": [
            "img2gcode=img2gcode.__main__:main",
        ]
    },
)
