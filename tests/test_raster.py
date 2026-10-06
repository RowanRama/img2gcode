import numpy as np
import pytest
from PIL import Image

from img2gcode.raster import (
    denoise, estimate_noise, fit_page, foreground_mask, load_rgb, luminance,
    mask_to_geometry, palette_kmeans, sample_colors, tone_levels, working_image,
)


def test_load_rgb_alpha(tmp_path):
    rgba = np.zeros((10, 10, 4), np.uint8)
    rgba[:5] = [255, 0, 0, 255]
    Image.fromarray(rgba, "RGBA").save(tmp_path / "a.png")
    rgb = load_rgb(tmp_path / "a.png")
    assert rgb.shape == (10, 10, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (255, 0, 0) and tuple(rgb[9, 9]) == (255, 255, 255)


def test_load_rgb_modes(tmp_path):
    base = np.zeros((10, 10, 3), np.uint8); base[:, 5:] = 200
    Image.fromarray(base).convert("P").save(tmp_path / "p.png")
    assert load_rgb(tmp_path / "p.png").shape == (10, 10, 3)
    g16 = np.linspace(0, 65535, 100).reshape(10, 10).astype(np.uint16)
    Image.fromarray(g16).save(tmp_path / "g16.png")
    g = load_rgb(tmp_path / "g16.png")
    assert g.shape == (10, 10, 3) and g.dtype == np.uint8
    assert g[0, 0, 0] < 20 and g[-1, -1, 0] > 235


def test_load_rgb_exif_orientation(tmp_path):
    img = Image.new("RGB", (40, 20), "white")
    exif = Image.Exif(); exif[0x0112] = 6
    img.save(tmp_path / "r.jpg", exif=exif)
    assert load_rgb(tmp_path / "r.jpg").shape[:2] == (40, 20)


def test_fit_page(cfg):
    w, h, s = fit_page(800, 600, cfg)
    assert s == pytest.approx(700 / 600)
    assert h == pytest.approx(700) and w == pytest.approx(800 * 700 / 600)


def test_working_image_resamples():
    rgb = np.zeros((600, 800, 3), np.uint8)
    img, mm = working_image(rgb, 1.25, 2.0)
    assert img.shape[:2] == (375, 500) and mm == pytest.approx(2.0, rel=0.01)
    up, mm2 = working_image(rgb, 1.25, 0.625)
    assert up.shape[1] == 1600 and mm2 == pytest.approx(0.625, rel=0.01)


def test_working_image_caps_size():
    rgb = np.zeros((4000, 6000, 3), np.uint8)
    img, mm = working_image(rgb, 1000 / 6000, 0.1, max_side=2500)
    assert max(img.shape[:2]) == 2500 and mm == pytest.approx(1000 / 2500, rel=0.01)


def test_estimate_noise():
    rng = np.random.default_rng(0)
    flat = np.full((200, 200), 128, np.uint8)
    assert estimate_noise(flat) < 0.5
    noisy = np.clip(128 + rng.normal(0, 20, (200, 200)), 0, 255).astype(np.uint8)
    assert 17 < estimate_noise(noisy) < 23


def test_denoise_reduces_noise():
    rng = np.random.default_rng(1)
    img = np.clip(128 + rng.normal(0, 25, (120, 160, 3)), 0, 255).astype(np.uint8)
    out = denoise(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    gray = lambda a: a.mean(axis=2).astype(np.uint8)
    assert estimate_noise(gray(out)) < 0.5 * estimate_noise(gray(img))
    clean = np.full((50, 50), 100, np.uint8)
    assert np.array_equal(denoise(clean), clean)


def test_luminance_and_foreground():
    rgb = np.array([[[0, 0, 0], [255, 255, 255], [230, 230, 230], [200, 255, 255]]], np.uint8)
    lum = luminance(rgb)
    assert lum.dtype == np.float32 and lum[0, 0] < 0.01 and lum[0, 1] > 0.99
    assert foreground_mask(rgb, 220).tolist() == [[True, False, False, True]]


def test_palette_kmeans_sorted_dark_to_light():
    img = np.zeros((30, 30, 3), np.uint8)
    img[:, 10:20] = (220, 30, 30); img[:, 20:] = (250, 240, 60)
    pal, labels = palette_kmeans(img, np.ones((30, 30), bool), 3)
    assert pal.shape == (3, 3) and pal.dtype == np.uint8
    assert tuple(pal[0]) == pytest.approx((0, 0, 0), abs=3)
    assert labels[0, 0] == 0 and labels[0, 15] == 1 and labels[0, 25] == 2


def test_palette_kmeans_more_clusters_than_colours():
    img = np.zeros((20, 20, 3), np.uint8); img[:, 10:] = 200
    mask = np.ones((20, 20), bool); mask[0, 0] = False
    pal, labels = palette_kmeans(img, mask, 5)
    assert 1 <= len(pal) <= 5 and labels.max() < len(pal) and labels[0, 0] == -1


def test_palette_kmeans_single_pixel():
    img = np.zeros((4, 4, 3), np.uint8); img[1, 2] = (200, 30, 30)
    mask = np.zeros((4, 4), bool); mask[1, 2] = True
    pal, labels = palette_kmeans(img, mask, 3)
    assert pal.shape == (1, 3) and tuple(pal[0]) == pytest.approx((200, 30, 30), abs=3)
    assert labels[1, 2] == 0 and (labels[~mask] == -1).all()


def test_palette_kmeans_empty_mask():
    pal, labels = palette_kmeans(np.zeros((5, 5, 3), np.uint8), np.zeros((5, 5), bool), 2)
    assert len(pal) == 0 and (labels == -1).all()


def test_mask_to_geometry_ring_and_island():
    yy, xx = np.mgrid[0:300, 0:300]
    r = np.hypot(xx - 150, yy - 150)
    ring = (r <= 100) & (r > 40)
    g = mask_to_geometry(ring, 1.0)
    assert g.area == pytest.approx(np.pi * (100 ** 2 - 40 ** 2), rel=0.02)
    assert len(list(g.interiors)) == 1
    island = ring | (r <= 20)
    g2 = mask_to_geometry(island, 0.5)
    assert g2.area == pytest.approx(0.25 * np.pi * (100 ** 2 - 40 ** 2 + 20 ** 2), rel=0.02)
    assert g2.geom_type == "MultiPolygon" and len(g2.geoms) == 2
    s = mask_to_geometry(ring, 1.0, simplify_mm=0.5)
    assert len(s.exterior.coords) < len(g.exterior.coords)
    assert mask_to_geometry(np.zeros((5, 5), bool), 1.0).is_empty


def test_mask_to_geometry_min_area():
    m = np.zeros((100, 100), bool); m[10:60, 10:60] = True; m[80:83, 80:83] = True; m[30:32, 30:32] = False
    g = mask_to_geometry(m, 1.0, min_area_mm2=20)
    assert g.geom_type == "Polygon" and len(list(g.interiors)) == 0


def test_tone_levels():
    lum = np.linspace(0, 1, 10001).reshape(1, -1).astype(np.float32)
    t = tone_levels(lum, np.ones_like(lum, bool), 4)
    assert np.allclose(t, [0.25, 0.5, 0.75], atol=1e-3)


def test_sample_colors():
    img = np.zeros((10, 10, 3), np.uint8); img[:, 5:] = (9, 8, 7)
    c = sample_colors(img, np.array([[1.0, 1.0], [8.0, 3.0]]), 1.0)
    assert c.tolist() == [[0, 0, 0], [9, 8, 7]]
