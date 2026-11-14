//! Text from any installed font, turned into fill-mode toolpaths.
//!
//! The glyph outlines come from the font itself (resvg lays the text out);
//! they are drawn into a fine mask and cut with the same smoothed distance
//! field as images, so text gets the same perimeters and, if wanted, infill.

use std::sync::{Arc, OnceLock};

use resvg::{tiny_skia, usvg};
use serde::{Deserialize, Serialize};

use crate::Error;
use crate::raster::Grid;
use crate::segment::BACKGROUND;

/// One block of text on the canvas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TextItem {
    /// The text; `\n` starts a new line.
    pub content: String,
    /// A font family ("Arial") or a face's full name ("Arial Black"), as
    /// `img2gcode fonts` lists them, or `sans-serif`, `serif`, `monospace`.
    pub font: String,
    /// Font size in mm (the em size; capitals are about 0.7 of it). The
    /// text is centred on its letters, not on its line box.
    pub size: f64,
    /// Centre of the text block on the machine (mm).
    pub x: f64,
    /// Centre of the text block on the machine (mm).
    pub y: f64,
    /// Bold face.
    pub bold: bool,
    /// Italic face.
    pub italic: bool,
    /// Fill the letters; `false` prints only their outlines.
    pub infill: bool,
    /// Tool that prints the text.
    pub tool: usize,
    /// Wrap words onto new lines to fit this width (mm); 0 = no wrapping.
    pub width: f64,
    /// How lines line up with each other.
    pub align: Align,
}

/// Line alignment within a text block.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Align {
    /// Lines start at the same x.
    Left,
    /// Lines are centred on each other.
    Center,
    /// Lines end at the same x.
    Right,
}

impl Default for TextItem {
    fn default() -> Self {
        Self {
            content: "Text".into(),
            font: "sans-serif".into(),
            size: 40.0,
            x: 0.0,
            y: 0.0,
            bold: false,
            italic: false,
            infill: true,
            tool: 0,
            width: 0.0,
            align: Align::Center,
        }
    }
}

/// The installed fonts, loaded once per process.
pub fn system_fonts() -> Arc<usvg::fontdb::Database> {
    static FONTS: OnceLock<Arc<usvg::fontdb::Database>> = OnceLock::new();
    FONTS
        .get_or_init(|| {
            let mut db = usvg::fontdb::Database::new();
            db.load_system_fonts();
            Arc::new(db)
        })
        .clone()
}

/// One installed font face under the name people know it by.
#[derive(Debug, Clone)]
struct Face {
    /// Full name from the font, e.g. "Arial Black", "Segoe UI Semibold Italic".
    name: String,
    /// Family it belongs to for layout, e.g. "Arial".
    family: String,
    weight: u16,
    italic: bool,
}

fn faces() -> &'static [Face] {
    static FACES: OnceLock<Vec<Face>> = OnceLock::new();
    FACES.get_or_init(|| {
        let db = system_fonts();
        db.faces()
            .filter_map(|f| {
                let family = f.families.first()?.0.clone();
                let name = db.with_face_data(f.id, full_name).flatten().unwrap_or_else(|| family.clone());
                Some(Face { name, family, weight: f.weight.0, italic: f.style != usvg::fontdb::Style::Normal })
            })
            .collect()
    })
}

/// Every name text can use: font families ("Arial") and the full names of
/// their faces ("Arial Black", "Arial Bold Italic"), sorted.
pub fn font_names() -> Vec<String> {
    let db = system_fonts();
    let mut v: Vec<String> = db.faces().flat_map(|f| f.families.iter().map(|(n, _)| n.clone())).collect();
    v.extend(faces().iter().map(|f| f.name.clone()));
    v.sort_by_key(|s| s.to_lowercase());
    v.dedup_by(|a, b| a.eq_ignore_ascii_case(b));
    v
}

const GENERIC: [&str; 5] = ["sans-serif", "serif", "monospace", "cursive", "fantasy"];

/// The family to lay out with, plus the weight and slant a full face name
/// fixes (`None` when `name` is a family and the item's bold/italic decide).
fn resolve(name: &str) -> Option<(String, Option<(u16, bool)>)> {
    let db = system_fonts();
    if GENERIC.iter().any(|g| g.eq_ignore_ascii_case(name))
        || db.faces().any(|f| f.families.iter().any(|(n, _)| n.eq_ignore_ascii_case(name)))
    {
        return Some((name.to_string(), None));
    }
    faces().iter().find(|f| f.name.eq_ignore_ascii_case(name)).map(|f| (f.family.clone(), Some((f.weight, f.italic))))
}

/// The font's full name (name record 4), preferring US English. `data` is
/// the font file, `index` the face within a collection. `None` on anything
/// malformed.
fn full_name(data: &[u8], index: u32) -> Option<String> {
    let u16_at = |o: usize| data.get(o..o.checked_add(2)?).map(|b| u16::from_be_bytes([b[0], b[1]]));
    let u32_at = |o: usize| data.get(o..o.checked_add(4)?).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    let base = if data.get(0..4)? == b"ttcf" {
        if index >= u32_at(8)? {
            return None;
        }
        u32_at(12 + 4 * index as usize)? as usize
    } else {
        0
    };
    let tables = u16_at(base + 4)? as usize;
    let name = (0..tables).find_map(|t| {
        let rec = base + 12 + 16 * t;
        (data.get(rec..rec + 4)? == b"name").then(|| u32_at(rec + 8)).flatten()
    })? as usize;
    let count = u16_at(name + 2)? as usize;
    let strings = name + u16_at(name + 4)? as usize;
    let mut fallback = None;
    for r in 0..count {
        let rec = name + 6 + 12 * r;
        let (platform, encoding, lang, id) = (u16_at(rec)?, u16_at(rec + 2)?, u16_at(rec + 4)?, u16_at(rec + 6)?);
        if id != 4 {
            continue;
        }
        let (len, off) = (u16_at(rec + 8)? as usize, u16_at(rec + 10)? as usize);
        let bytes = data.get(strings + off..strings + off + len)?;
        let text = match (platform, encoding) {
            (0, _) | (3, 0 | 1 | 10) => {
                let units: Vec<u16> = bytes.as_chunks::<2>().0.iter().map(|c| u16::from_be_bytes(*c)).collect();
                String::from_utf16_lossy(&units)
            }
            (1, 0) => bytes.iter().map(|&b| b as char).collect(),
            _ => continue,
        };
        if platform == 3 && lang == 0x409 {
            return Some(text);
        }
        fallback.get_or_insert(text);
    }
    fallback
}

/// A text item drawn as a mask, with what is needed to put its pixels back
/// on the machine.
#[derive(Debug, Clone)]
pub struct TextMask {
    /// 0 inside the letters, [`BACKGROUND`] outside.
    pub labels: Grid<u8>,
    /// Pixels per mm.
    pub ppm: f64,
    /// Machine position of pixel centre (0, 0).
    pub origin: [f64; 2],
    /// Machine bounds of the text `[xmin, ymin, xmax, ymax]`.
    pub bounds: [f64; 4],
}

impl TextMask {
    /// Machine position of a point in pixel coordinates (pixel centres at
    /// integers, Y down).
    pub fn to_mm(&self, p: [f64; 2]) -> [f64; 2] {
        [self.origin[0] + p[0] / self.ppm, self.origin[1] - p[1] / self.ppm]
    }
}

fn escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// Advance widths (mm) of each string set in `style`, measured in one layout.
fn measure(strings: &[&str], style: &str, opt: &usvg::Options) -> Result<Vec<f64>, Error> {
    let mut svg = String::from(r#"<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1">"#);
    for (i, s) in strings.iter().enumerate() {
        svg.push_str(&format!(r#"<text y="{}" {style}>{}</text>"#, i * 1000, escape(s)));
    }
    svg.push_str("</svg>");
    let tree = usvg::Tree::from_str(&svg, opt).map_err(|e| Error::Svg(e.to_string()))?;
    Ok(tree
        .root()
        .children()
        .iter()
        .filter_map(|n| if let usvg::Node::Text(t) = n { Some(t.bounding_box().width() as f64) } else { None })
        .collect())
}

/// Break `content` into lines no wider than `width` (mm), at spaces. A word
/// wider than the box gets a line of its own. `\n` still forces a break.
fn wrap(content: &str, width: f64, style: &str, opt: &usvg::Options) -> Result<Vec<String>, Error> {
    let mut words: Vec<&str> = content.split_whitespace().collect();
    words.sort_unstable();
    words.dedup();
    // A space's width is what it adds between two letters.
    let mut probe = words.clone();
    probe.extend(["x x", "xx"]);
    let widths = measure(&probe, style, opt)?;
    if widths.len() != probe.len() {
        // Some string drew nothing, so the widths cannot be matched up.
        return Ok(content.lines().map(String::from).collect());
    }
    let space = (widths[widths.len() - 2] - widths[widths.len() - 1]).max(0.0);
    let width_of = |w: &str| words.binary_search(&w).map_or(0.0, |i| widths[i]);
    let mut lines = Vec::new();
    for para in content.lines() {
        let (mut line, mut used) = (String::new(), 0.0);
        for word in para.split_whitespace() {
            let w = width_of(word);
            if line.is_empty() {
                (line, used) = (word.to_string(), w);
            } else if used + space + w <= width {
                line.push(' ');
                line.push_str(word);
                used += space + w;
            } else {
                lines.push(std::mem::replace(&mut line, word.to_string()));
                used = w;
            }
        }
        lines.push(line);
    }
    Ok(lines)
}

/// Lay out and draw `item`. `None` when it has no visible glyphs.
/// `line_width` (mm) sets how finely it is drawn: at least 6 px per line.
pub fn render(item: &TextItem, line_width: f64) -> Result<Option<TextMask>, Error> {
    if item.content.trim().is_empty() || item.size <= 0.0 {
        return Ok(None);
    }
    let fonts = system_fonts();
    let (family, face) = resolve(&item.font).ok_or_else(|| Error::Font(item.font.clone()))?;
    let (weight, italic) = match face {
        Some((w, i)) => (w, i || item.italic),
        None => (if item.bold { 700 } else { 400 }, item.italic),
    };
    // One SVG unit = 1 mm.
    let style = format!(
        r#"font-family="{}" font-size="{}" font-weight="{}" font-style="{}""#,
        escape(&family),
        item.size,
        weight,
        if italic { "italic" } else { "normal" },
    );
    let opt = usvg::Options { fontdb: fonts, ..Default::default() };
    let lines = if item.width > 0.0 {
        wrap(&item.content, item.width, &style, &opt)?
    } else {
        item.content.lines().map(String::from).collect()
    };
    let mut spans = String::new();
    for (i, line) in lines.iter().enumerate() {
        let dy = if i == 0 { "0" } else { "1.2em" };
        // An empty line still takes its height.
        let text = if line.is_empty() { "\u{a0}" } else { line };
        spans.push_str(&format!(r#"<tspan x="0" dy="{dy}">{}</tspan>"#, escape(text)));
    }
    let anchor = match item.align {
        Align::Left => "start",
        Align::Center => "middle",
        Align::Right => "end",
    };
    let svg = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"><text {style} text-anchor="{anchor}" fill="#000">{spans}</text></svg>"##
    );
    let tree = usvg::Tree::from_str(&svg, &opt).map_err(|e| Error::Svg(e.to_string()))?;
    // The layout box (line height, advance widths), not the ink: glyphs can
    // overhang it, so draw a margin around it and measure the ink after.
    let Some(bbox) = tree.root().abs_bounding_box() else { return Ok(None) };
    let margin = 0.3 * item.size;
    let (bx, by) = (bbox.x() as f64 - margin, bbox.y() as f64 - margin);
    let (bw, bh) = (bbox.width() as f64 + 2.0 * margin, bbox.height() as f64 + 2.0 * margin);
    // Fine enough for a smooth outline, capped at 4000 px on the long side.
    let ppm = (6.0 / line_width.max(1e-3)).max(10.0).min(4000.0 / bw.max(bh));
    let (w, h) = ((bw * ppm).ceil() as u32, (bh * ppm).ceil() as u32);
    let mut pm = tiny_skia::Pixmap::new(w, h).ok_or_else(|| Error::Svg("text too large to draw".into()))?;
    let t = tiny_skia::Transform::from_row(ppm as f32, 0.0, 0.0, ppm as f32, (-bx * ppm) as f32, (-by * ppm) as f32);
    resvg::render(&tree, t, &mut pm.as_mut());
    let (w, h) = (w as usize, h as usize);
    let ink: Vec<bool> = pm.pixels().iter().map(|p| p.alpha() >= 128).collect();
    let (mut i0, mut j0, mut i1, mut j1) = (usize::MAX, usize::MAX, 0, 0);
    for (k, _) in ink.iter().enumerate().filter(|(_, b)| **b) {
        let (i, j) = (k % w, k / w);
        (i0, j0, i1, j1) = (i0.min(i), j0.min(j), i1.max(i), j1.max(j));
    }
    if i0 > i1 {
        return Ok(None);
    }
    // Crop to the ink plus a few empty pixels.
    let pad = 4;
    let (ci, cj) = (i0.saturating_sub(pad), j0.saturating_sub(pad));
    let (cw, ch) = ((i1 + pad + 1).min(w) - ci, (j1 + pad + 1).min(h) - cj);
    let mut labels = Grid::new(cw, ch, BACKGROUND);
    for y in 0..ch {
        for x in 0..cw {
            if ink[(y + cj) * w + x + ci] {
                labels.set(x, y, 0);
            }
        }
    }
    // Full-image pixel centre (i, j) is layout point bx + (i + 0.5) / ppm.
    // The ink's centre goes to (item.x, item.y), Y flipped.
    let (iw, ih) = ((i1 + 1 - i0) as f64 / ppm, (j1 + 1 - j0) as f64 / ppm);
    let (cx, cy) = (bx + i0 as f64 / ppm + iw / 2.0, by + j0 as f64 / ppm + ih / 2.0);
    let origin = [item.x + (bx + (ci as f64 + 0.5) / ppm - cx), item.y - (by + (cj as f64 + 0.5) / ppm - cy)];
    let bounds = [item.x - iw / 2.0, item.y - ih / 2.0, item.x + iw / 2.0, item.y + ih / 2.0];
    Ok(Some(TextMask { labels, ppm, origin, bounds }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn some_font() -> Option<String> {
        font_names().into_iter().find(|f| ["Arial", "DejaVu Sans", "Liberation Sans"].contains(&f.as_str()))
    }

    #[test]
    fn text_is_centred_on_its_position_and_sized_in_mm() {
        let Some(font) = some_font() else { return };
        let item = TextItem { content: "HHHH".into(), font, size: 50.0, x: 100.0, y: -20.0, ..Default::default() };
        let m = render(&item, 1.0).unwrap().unwrap();
        let b = m.bounds;
        assert!(((b[0] + b[2]) / 2.0 - 100.0).abs() < 1e-6 && ((b[1] + b[3]) / 2.0 + 20.0).abs() < 1e-6);
        // Arial capitals are 0.716 em tall.
        assert!((b[3] - b[1] - 35.8).abs() < 2.0, "height {}", b[3] - b[1]);
        // Inked pixels land inside the bounds.
        let inked: Vec<[f64; 2]> = (0..m.labels.h)
            .flat_map(|y| (0..m.labels.w).map(move |x| (x, y)))
            .filter(|&(x, y)| m.labels.get(x, y) == 0)
            .map(|(x, y)| m.to_mm([x as f64, y as f64]))
            .collect();
        assert!(!inked.is_empty());
        let tol = 1.0 / m.ppm + 1e-9;
        for p in inked {
            assert!(p[0] >= b[0] - tol && p[0] <= b[2] + tol && p[1] >= b[1] - tol && p[1] <= b[3] + tol, "{p:?}");
        }
    }

    #[test]
    fn unknown_font_is_an_error_and_blank_text_is_nothing() {
        let item = TextItem { font: "No Such Font 12345".into(), ..Default::default() };
        assert!(matches!(render(&item, 1.0), Err(Error::Font(_))));
        let blank = TextItem { content: "  ".into(), ..Default::default() };
        assert!(render(&blank, 1.0).unwrap().is_none());
    }

    #[test]
    fn full_name_reads_a_name_table_and_refuses_garbage() {
        // A minimal sfnt: one table ("name") holding one Windows US-English
        // full-name record.
        let s: Vec<u8> = "Demo Bold".encode_utf16().flat_map(u16::to_be_bytes).collect();
        let mut f = vec![0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0];
        f.extend_from_slice(b"name");
        f.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 28, 0, 0, 0, 0]);
        // name table at 28: format 0, 1 record, strings at 6 + 12.
        f.extend_from_slice(&[0, 0, 0, 1, 0, 18]);
        f.extend_from_slice(&[0, 3, 0, 1, 0x04, 0x09, 0, 4, 0, s.len() as u8, 0, 0]);
        f.extend_from_slice(&s);
        assert_eq!(full_name(&f, 0).as_deref(), Some("Demo Bold"));
        assert_eq!(full_name(&f[..30], 0), None);
        assert_eq!(full_name(b"ttcf\0\0\0\0\0\0\0\x05", 9), None);
        assert_eq!(full_name(&[], 0), None);
    }

    #[test]
    fn a_full_face_name_selects_that_face() {
        if !font_names().iter().any(|n| n == "Arial Black") {
            return;
        }
        let (family, face) = resolve("arial black").unwrap();
        assert_eq!(family, "Arial");
        assert_eq!(face, Some((900, false)));
        let regular = render(&TextItem { content: "H".into(), font: "Arial".into(), ..Default::default() }, 1.0);
        let black = render(&TextItem { content: "H".into(), font: "Arial Black".into(), ..Default::default() }, 1.0);
        let ink = |m: TextMask| m.labels.data.iter().filter(|&&l| l == 0).count() as f64 / (m.ppm * m.ppm);
        assert!(ink(black.unwrap().unwrap()) > 1.3 * ink(regular.unwrap().unwrap()));
    }

    #[test]
    fn wrapping_keeps_lines_inside_the_width() {
        let Some(font) = some_font() else { return };
        let content = "one two three four five six seven eight nine ten".to_string();
        let one_line = render(&TextItem { content: content.clone(), font: font.clone(), ..Default::default() }, 1.0)
            .unwrap()
            .unwrap();
        let wrapped = render(&TextItem { content, font, width: 200.0, ..Default::default() }, 1.0).unwrap().unwrap();
        let w = |m: &TextMask| m.bounds[2] - m.bounds[0];
        let h = |m: &TextMask| m.bounds[3] - m.bounds[1];
        assert!(w(&one_line) > 400.0);
        assert!(w(&wrapped) <= 200.0 + 1.0, "wrapped width {}", w(&wrapped));
        assert!(h(&wrapped) > 2.5 * h(&one_line), "only {} high", h(&wrapped));
    }

    #[test]
    fn a_word_longer_than_the_width_keeps_its_own_line() {
        let opt = usvg::Options { fontdb: system_fonts(), ..Default::default() };
        let lines = wrap("a extraordinarily b", 5.0, r#"font-family="sans-serif" font-size="10""#, &opt).unwrap();
        assert_eq!(lines, vec!["a", "extraordinarily", "b"]);
    }

    #[test]
    fn lines_stack_downwards() {
        let Some(font) = some_font() else { return };
        let one =
            render(&TextItem { content: "A".into(), font: font.clone(), ..Default::default() }, 1.0).unwrap().unwrap();
        let two = render(&TextItem { content: "A\nA".into(), font, ..Default::default() }, 1.0).unwrap().unwrap();
        assert!(two.bounds[3] - two.bounds[1] > 1.8 * (one.bounds[3] - one.bounds[1]));
    }
}
