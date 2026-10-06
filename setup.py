from setuptools import setup, find_packages

setup(
    name="img2gcode",
    version="0.2.0",
    description="Convert images to SVG toolpaths and multi-tool GCode",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "Pillow>=10.0",
        "opencv-python-headless>=4.8",
        "matplotlib>=3.7",
        "svgelements>=1.9",
        "shapely>=2.0",
        "scikit-image>=0.22",
        "scipy>=1.10",
    ],
    entry_points={
        "console_scripts": [
            "img2gcode=img2gcode.__main__:main",
        ]
    },
)
