"""Image segmentation pipeline.

Steps
-----
1. Load image as RGBA → convert to RGB.
2. Mask out near-white background pixels.
3. Run K-Means on the foreground pixels to assign each to one of *n* tools.
4. Return a label map (HxW int array) where:
   - -1  → background (white / transparent)
   -  0…n-1 → tool index
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


def show_debug(
    rgb: np.ndarray,
    fg_mask: np.ndarray,
    label_map: np.ndarray,
    cluster_colors: np.ndarray,
) -> None:
    """Display original, background-removed, and k-means results side by side."""
    import matplotlib.pyplot as plt

    after_bg = rgb.copy()
    after_bg[~fg_mask] = 0  # black out background

    H, W = rgb.shape[:2]
    kmeans_img = np.full((H, W, 3), 255, dtype=np.uint8)
    for i, color in enumerate(cluster_colors):
        kmeans_img[label_map == i] = color

    # Flip vertically so orientation matches the GCode visualiser (Y increases upward)
    _, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(np.flipud(rgb))
    axes[0].set_title("Original")
    axes[1].imshow(np.flipud(after_bg))
    axes[1].set_title("After background removal")
    axes[2].imshow(np.flipud(kmeans_img))
    axes[2].set_title(f"After K-Means ({len(cluster_colors)} clusters)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def load_image(path: str) -> np.ndarray:
    """Load any PIL-supported image and return an (H, W, 3) uint8 RGB array."""
    img = Image.open(path).convert("RGBA")
    # Composite onto white background to handle transparency
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))
    background.paste(img, mask=img.split()[3])
    rgb = background.convert("RGB")
    return np.asarray(rgb, dtype=np.uint8)


def remove_background(
    rgb: np.ndarray, white_threshold: int = 220
) -> np.ndarray:
    """Return boolean mask (H, W) – True where pixel is *foreground*."""
    return ~np.all(rgb >= white_threshold, axis=-1)


def segment(
    image_path: str,
    n_tools: int = 2,
    white_threshold: int = 220,
    min_cluster_area: int = 50,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment an image into *n_tools* tool regions.

    Returns
    -------
    label_map : (H, W) int array — -1 = background, 0…n-1 = tool index
    cluster_colors : (n_tools, 3) uint8 — representative RGB colour per tool
    fg_mask : (H, W) bool — foreground mask
    """
    rgb = load_image(image_path)
    fg_mask = remove_background(rgb, white_threshold)

    foreground_pixels = rgb[fg_mask]  # (N, 3)

    if len(foreground_pixels) == 0:
        raise ValueError(
            "No foreground pixels found. Try lowering white_threshold in the config."
        )

    # K-Means clustering on foreground pixels
    n_clusters = min(n_tools, len(foreground_pixels))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    km.fit(foreground_pixels)

    labels_flat = km.labels_  # (N,)
    cluster_colors = km.cluster_centers_.astype(np.uint8)  # (k, 3)

    # Build full label map
    H, W = rgb.shape[:2]
    label_map = np.full((H, W), fill_value=-1, dtype=np.int32)
    label_map[fg_mask] = labels_flat

    # Per-tool connected-component speckle filter:
    # the previous behaviour only checked total tool area, so anti-aliased pixels
    # near a logo edge formed many tiny spurious blobs that all survived. Now we
    # drop each connected component below min_cluster_area independently.
    label_map = _remove_small_components(label_map, n_clusters, min_cluster_area)

    # show_debug(rgb, fg_mask, label_map, cluster_colors)

    return label_map, cluster_colors, fg_mask


def _remove_small_components(
    label_map: np.ndarray, n_clusters: int, min_area: int
) -> np.ndarray:
    """Drop connected components smaller than *min_area* px from each tool.

    Cleared pixels become background (-1).  Operates per-tool so that tiny
    speckles of one tool sitting next to a large region of another tool are
    eliminated independently.
    """
    if min_area <= 0:
        return label_map

    out = label_map.copy()
    for t in range(n_clusters):
        mask = (label_map == t).astype(np.uint8)
        if mask.sum() == 0:
            continue
        n_comp, comp_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # Component 0 is the background of the per-tool mask; skip it.
        for cidx in range(1, n_comp):
            if stats[cidx, cv2.CC_STAT_AREA] < min_area:
                out[comp_labels == cidx] = -1
    return out
