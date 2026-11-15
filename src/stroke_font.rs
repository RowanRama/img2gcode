//! Single-line (stroke) fonts: every letter is a few open strokes for the
//! tool to follow once, instead of an outline to fill.
//!
//! They come as SVG fonts, the format Inkscape's Hershey Text uses. Relief
//! SingleLine is built in; the Hershey fonts of an Inkscape install are
//! found too, and any other SVG font can be named by its file path.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use svgtypes::{SimplePathSegment, SimplifyingPathParser};

use crate::Error;
use crate::geom::Pt;
use crate::text::{Align, TextItem};

/// The built-in font's name.
pub const RELIEF: &str = "Relief SingleLine";
const RELIEF_SVG: &str = include_str!("../fonts/ReliefSingleLineSVG-Regular.svg");

/// Lines are this many em apart, like outline text.
const LINE_HEIGHT: f64 = 1.2;
/// Slant of italic text (x shift per unit of height).
const SLANT: f64 = 0.2;

/// One glyph in font units, Y up from the baseline.
#[derive(Debug, Clone, Default)]
struct Glyph {
    advance: f64,
    strokes: Vec<Vec<Pt>>,
}

/// A parsed SVG font.
#[derive(Debug, Clone)]
pub struct StrokeFont {
    /// The name text items use to pick it.
    pub name: String,
    units_per_em: f64,
    missing_advance: f64,
    glyphs: HashMap<char, Glyph>,
    kerning: HashMap<(char, char), f64>,
}

impl StrokeFont {
    /// Parse the first `<font>` of an SVG font file. `name` replaces its
    /// family name when given.
    pub fn parse(svg: &str, name: Option<&str>) -> Result<Self, Error> {
        let bad = |why: &str| Error::Font(format!("SVG font: {why}"));
        // FontForge writes a DOCTYPE.
        let opt = roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() };
        let doc = roxmltree::Document::parse_with_options(svg, opt).map_err(|e| bad(&e.to_string()))?;
        let font = doc.descendants().find(|n| n.has_tag_name("font")).ok_or_else(|| bad("no <font> element"))?;
        let num = |n: roxmltree::Node, a: &str| n.attribute(a).and_then(|v| v.trim().parse::<f64>().ok());
        let face = font.children().find(|n| n.has_tag_name("font-face"));
        let units_per_em = face.and_then(|f| num(f, "units-per-em")).filter(|u| *u > 0.0).unwrap_or(1000.0);
        let default_advance = num(font, "horiz-adv-x").unwrap_or(units_per_em / 2.0);
        let name = name
            .map(String::from)
            .or_else(|| face.and_then(|f| f.attribute("font-family")).map(String::from))
            .or_else(|| font.attribute("id").map(String::from))
            .unwrap_or_else(|| "SVG font".into());
        let missing_advance = font
            .children()
            .find(|n| n.has_tag_name("missing-glyph"))
            .and_then(|n| num(n, "horiz-adv-x"))
            .unwrap_or(default_advance);

        // Curves are flattened to about 1/1000 em.
        let tolerance = units_per_em / 1000.0;
        let mut glyphs = HashMap::new();
        let mut by_name: HashMap<&str, char> = HashMap::new();
        for g in font.children().filter(|n| n.has_tag_name("glyph")) {
            // Ligatures (several characters) are left out.
            let mut chars = g.attribute("unicode").unwrap_or("").chars();
            let (Some(c), None) = (chars.next(), chars.next()) else { continue };
            if let Some(n) = g.attribute("glyph-name") {
                by_name.insert(n, c);
            }
            let strokes = g.attribute("d").map(|d| flatten(d, tolerance)).unwrap_or_default();
            glyphs.insert(c, Glyph { advance: num(g, "horiz-adv-x").unwrap_or(default_advance), strokes });
        }

        let mut kerning = HashMap::new();
        for k in font.children().filter(|n| n.has_tag_name("hkern")) {
            let Some(amount) = num(k, "k") else { continue };
            let side = |u: &str, g: &str| -> Vec<char> {
                let mut v: Vec<char> = k.attribute(u).map(unicode_list).unwrap_or_default();
                v.extend(k.attribute(g).into_iter().flat_map(|s| s.split(',')).filter_map(|n| by_name.get(n.trim())));
                v
            };
            for a in side("u1", "g1") {
                for b in side("u2", "g2") {
                    kerning.insert((a, b), amount);
                }
            }
        }
        Ok(Self { name, units_per_em, missing_advance, glyphs, kerning })
    }

    fn advance(&self, c: char) -> f64 {
        self.glyphs.get(&c).map_or(self.missing_advance, |g| g.advance)
    }

    /// Width of `s` in font units, kerning included.
    fn width(&self, s: &str) -> f64 {
        let mut w = 0.0;
        let mut prev = None;
        for c in s.chars() {
            if let Some(p) = prev {
                w -= self.kerning.get(&(p, c)).copied().unwrap_or(0.0);
            }
            w += self.advance(c);
            prev = Some(c);
        }
        w
    }

    /// Lay out `item` as strokes on the machine (mm, Y up), centred on the
    /// item's position by their extent. Also returns that extent
    /// `[xmin, ymin, xmax, ymax]`. `None` when nothing would be drawn.
    pub fn layout(&self, item: &TextItem) -> Option<(Vec<Vec<Pt>>, [f64; 4])> {
        if item.content.trim().is_empty() || item.size <= 0.0 {
            return None;
        }
        let em = self.units_per_em;
        let lines = if item.width > 0.0 {
            wrap(&item.content, item.width / item.size * em, |s| self.width(s))
        } else {
            item.content.lines().map(String::from).collect()
        };
        let slant = if item.italic { SLANT } else { 0.0 };
        let mut strokes = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let w = self.width(line);
            let mut x = match item.align {
                Align::Left => 0.0,
                Align::Center => -w / 2.0,
                Align::Right => -w,
            };
            let base = -(i as f64) * LINE_HEIGHT * em;
            let mut prev = None;
            for c in line.chars() {
                if let Some(p) = prev {
                    x -= self.kerning.get(&(p, c)).copied().unwrap_or(0.0);
                }
                if let Some(g) = self.glyphs.get(&c) {
                    for s in &g.strokes {
                        strokes.push(s.iter().map(|p| [x + p[0] + slant * p[1], base + p[1]]).collect::<Vec<Pt>>());
                    }
                }
                x += self.advance(c);
                prev = Some(c);
            }
        }
        let mut b = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for p in strokes.iter().flatten() {
            b = [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])];
        }
        if !b[0].is_finite() {
            return None;
        }
        let s = item.size / em;
        let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
        let to_mm = |p: &Pt| -> Pt { [item.x + (p[0] - cx) * s, item.y + (p[1] - cy) * s] };
        let strokes = strokes.iter().map(|st| st.iter().map(to_mm).collect()).collect();
        let (hw, hh) = ((b[2] - b[0]) * s / 2.0, (b[3] - b[1]) * s / 2.0);
        Some((strokes, [item.x - hw, item.y - hh, item.x + hw, item.y + hh]))
    }
}

/// `u1`/`u2` of a kerning pair: comma-separated characters or `U+XXXX`.
fn unicode_list(s: &str) -> Vec<char> {
    s.split(',')
        .filter_map(|t| {
            let t = t.trim();
            match t.strip_prefix("U+").or_else(|| t.strip_prefix("u+")) {
                Some(hex) => u32::from_str_radix(hex, 16).ok().and_then(char::from_u32),
                None => {
                    let mut c = t.chars();
                    match (c.next(), c.next()) {
                        (Some(ch), None) => Some(ch),
                        _ => None,
                    }
                }
            }
        })
        .collect()
}

/// Path data to polylines, curves cut into chords within `tol`. A closed
/// subpath ends on its first point.
fn flatten(d: &str, tol: f64) -> Vec<Vec<Pt>> {
    let mut out: Vec<Vec<Pt>> = Vec::new();
    let mut cur: Vec<Pt> = Vec::new();
    let mut last: Pt = [0.0, 0.0];
    let finish = |cur: &mut Vec<Pt>, out: &mut Vec<Vec<Pt>>| {
        if cur.len() >= 2 {
            out.push(std::mem::take(cur));
        } else {
            cur.clear();
        }
    };
    // Enough chords that each strays at most `tol` from the curve, judged
    // from the control polygon's length.
    let steps = |len: f64| ((len / (8.0 * tol).max(1e-9)).sqrt().ceil() as usize).clamp(1, 64);
    for seg in SimplifyingPathParser::from(d).flatten() {
        match seg {
            SimplePathSegment::MoveTo { x, y } => {
                finish(&mut cur, &mut out);
                last = [x, y];
                cur.push(last);
            }
            SimplePathSegment::LineTo { x, y } => {
                if cur.is_empty() {
                    cur.push(last);
                }
                last = [x, y];
                cur.push(last);
            }
            SimplePathSegment::Quadratic { x1, y1, x, y } => {
                if cur.is_empty() {
                    cur.push(last);
                }
                let (p0, p1, p2) = (last, [x1, y1], [x, y]);
                let n = steps(crate::geom::length(&[p0, p1, p2]) * 4.0);
                for k in 1..=n {
                    let t = k as f64 / n as f64;
                    let u = 1.0 - t;
                    cur.push([
                        u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
                        u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
                    ]);
                }
                last = p2;
            }
            SimplePathSegment::CurveTo { x1, y1, x2, y2, x, y } => {
                if cur.is_empty() {
                    cur.push(last);
                }
                let (p0, p1, p2, p3) = (last, [x1, y1], [x2, y2], [x, y]);
                let n = steps(crate::geom::length(&[p0, p1, p2, p3]) * 4.0);
                for k in 1..=n {
                    let t = k as f64 / n as f64;
                    let u = 1.0 - t;
                    let (a, b, c, e) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                    cur.push([
                        a * p0[0] + b * p1[0] + c * p2[0] + e * p3[0],
                        a * p0[1] + b * p1[1] + c * p2[1] + e * p3[1],
                    ]);
                }
                last = p3;
            }
            SimplePathSegment::ClosePath => {
                if let Some(&first) = cur.first() {
                    if cur.last() != Some(&first) {
                        cur.push(first);
                    }
                    last = first;
                }
                finish(&mut cur, &mut out);
            }
        }
    }
    finish(&mut cur, &mut out);
    out
}

/// Break `content` into lines no wider than `width` at spaces, measuring
/// with `width_of`. A word wider than the box keeps a line of its own.
fn wrap(content: &str, width: f64, width_of: impl Fn(&str) -> f64) -> Vec<String> {
    let mut lines = Vec::new();
    for para in content.lines() {
        let mut line = String::new();
        for word in para.split_whitespace() {
            if line.is_empty() {
                line = word.to_string();
                continue;
            }
            let candidate = format!("{line} {word}");
            if width_of(&candidate) <= width {
                line = candidate;
            } else {
                lines.push(std::mem::replace(&mut line, word.to_string()));
            }
        }
        lines.push(line);
    }
    lines
}

/// Folders an Inkscape install keeps its Hershey SVG fonts in.
fn inkscape_dirs() -> Vec<PathBuf> {
    let mut v = vec![
        PathBuf::from(r"C:\Program Files\Inkscape\share\inkscape\extensions\svg_fonts"),
        PathBuf::from("/usr/share/inkscape/extensions/svg_fonts"),
        PathBuf::from("/usr/local/share/inkscape/extensions/svg_fonts"),
        PathBuf::from("/Applications/Inkscape.app/Contents/Resources/share/inkscape/extensions/svg_fonts"),
    ];
    if let Some(home) = std::env::var_os("HOME") {
        v.push(Path::new(&home).join(".config/inkscape/extensions/svg_fonts"));
    }
    v
}

/// Every stroke font: the built-in one, then any Inkscape ones, loaded once.
pub fn fonts() -> &'static [StrokeFont] {
    static FONTS: OnceLock<Vec<StrokeFont>> = OnceLock::new();
    FONTS.get_or_init(|| {
        // The built-in font is part of the build; a test checks it parses.
        let mut v: Vec<StrokeFont> = StrokeFont::parse(RELIEF_SVG, Some(RELIEF)).into_iter().collect();
        for dir in inkscape_dirs() {
            let Ok(entries) = std::fs::read_dir(dir) else { continue };
            let mut files: Vec<PathBuf> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("svg")))
                .collect();
            files.sort();
            for f in files {
                let Ok(s) = std::fs::read_to_string(&f) else { continue };
                if let Ok(font) = StrokeFont::parse(&s, None)
                    && !v.iter().any(|o| o.name.eq_ignore_ascii_case(&font.name))
                {
                    v.push(font);
                }
            }
        }
        v
    })
}

/// Names of every stroke font, built-in first.
pub fn names() -> Vec<String> {
    fonts().iter().map(|f| f.name.clone()).collect()
}

/// Whether `name` picks a stroke font, without reading any file.
pub fn is_stroke(name: &str) -> bool {
    let name = name.trim();
    fonts().iter().any(|f| f.name.eq_ignore_ascii_case(name))
        || Path::new(name).extension().is_some_and(|e| e.eq_ignore_ascii_case("svg"))
}

/// The stroke font `name` picks: one of [`fonts`], or an SVG font file
/// path. `Ok(None)` when `name` is not a stroke font (it may be an outline
/// font).
pub fn find(name: &str) -> Result<Option<std::borrow::Cow<'static, StrokeFont>>, Error> {
    if let Some(f) = fonts().iter().find(|f| f.name.eq_ignore_ascii_case(name.trim())) {
        return Ok(Some(std::borrow::Cow::Borrowed(f)));
    }
    let path = Path::new(name.trim());
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("svg")) {
        let svg = std::fs::read_to_string(path).map_err(|e| Error::Io(path.display().to_string(), e))?;
        return StrokeFont::parse(&svg, None).map(|f| Some(std::borrow::Cow::Owned(f)));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY: &str = r#"<svg xmlns="http://www.w3.org/2000/svg"><defs>
        <font id="Tiny" horiz-adv-x="500">
          <font-face font-family="Tiny Stroke" units-per-em="1000"/>
          <missing-glyph horiz-adv-x="300"/>
          <glyph unicode="I" horiz-adv-x="200" d="M100 0v700"/>
          <glyph unicode="L" glyph-name="L" horiz-adv-x="400" d="M50 700V0h300"/>
          <glyph unicode="O" horiz-adv-x="600" d="M300 0c150 0 250 150 250 350s-100 350 -250 350s-250 -150 -250 -350s100 -350 250 -350z"/>
          <glyph unicode=" " horiz-adv-x="250"/>
          <glyph unicode="fi" d="M0 0h10"/>
          <hkern g1="L" u2="I" k="100"/>
        </font></defs></svg>"#;

    fn item(content: &str) -> TextItem {
        TextItem { content: content.into(), font: "Tiny Stroke".into(), size: 10.0, ..TextItem::default() }
    }

    #[test]
    fn glyphs_are_open_strokes_not_outlines() {
        let f = StrokeFont::parse(TINY, None).unwrap();
        assert_eq!(f.name, "Tiny Stroke");
        let l = &f.glyphs[&'L'];
        assert_eq!(l.strokes, vec![vec![[50.0, 700.0], [50.0, 0.0], [350.0, 0.0]]]);
        // A closed subpath ends where it started.
        let o = &f.glyphs[&'O'].strokes[0];
        assert_eq!(o.first(), o.last());
        assert!(o.len() > 16, "curves are flattened");
        assert!(!f.glyphs.contains_key(&'f'), "ligatures are skipped");
    }

    #[test]
    fn layout_centres_on_the_strokes_in_mm() {
        let f = StrokeFont::parse(TINY, None).unwrap();
        let mut it = item("I");
        it.x = 20.0;
        it.y = -5.0;
        let (strokes, b) = f.layout(&it).unwrap();
        // 700 units at 10 mm per 1000 units = 7 mm tall, centred on (20, -5).
        assert_eq!(strokes, vec![vec![[20.0, -8.5], [20.0, -1.5]]]);
        assert_eq!(b, [20.0, -8.5, 20.0, -1.5]);
    }

    #[test]
    fn advances_and_kerning_place_letters() {
        let f = StrokeFont::parse(TINY, None).unwrap();
        assert_eq!(f.width("LI"), 400.0 - 100.0 + 200.0);
        assert_eq!(f.width("I?"), 200.0 + 300.0, "unknown characters take the missing-glyph advance");
        let (s, _) = f.layout(&TextItem { align: Align::Left, ..item("LI") }).unwrap();
        // The I's stroke is 300 units right of the L's stem start + 50.
        let dx = s[1][0][0] - s[0][0][0];
        assert!((dx - 3.5).abs() < 1e-9, "{dx}");
    }

    #[test]
    fn lines_stack_and_wrap_at_spaces() {
        let f = StrokeFont::parse(TINY, None).unwrap();
        let (one, _) = f.layout(&item("I\nI")).unwrap();
        assert!((one[0][0][1] - one[1][0][1] - 12.0).abs() < 1e-9, "1.2 em apart");
        // "I I I" is 200+250+200+250+200 units; 7 mm (700 units) fits "I I".
        let lines = wrap("I I I", 700.0, |s| f.width(s));
        assert_eq!(lines, vec!["I I", "I"]);
        let (wrapped, b) = f.layout(&TextItem { width: 7.0, ..item("I I I") }).unwrap();
        assert_eq!(wrapped.len(), 3);
        assert!((b[3] - b[1] - 19.0).abs() < 1e-9, "two lines: 7 mm + 12 mm");
    }

    #[test]
    fn italic_slants_and_blank_text_is_nothing() {
        let f = StrokeFont::parse(TINY, None).unwrap();
        let (s, _) = f.layout(&TextItem { italic: true, ..item("I") }).unwrap();
        assert!(s[0][1][0] > s[0][0][0], "the top leans right");
        assert!(f.layout(&item("  ")).is_none());
        assert!(f.layout(&item("?")).is_none(), "no strokes, nothing drawn");
    }

    #[test]
    fn relief_is_built_in_and_findable() {
        let names = names();
        assert_eq!(names[0], RELIEF);
        let f = find("relief singleline").unwrap().unwrap();
        let (strokes, b) = f.layout(&TextItem { content: "Rowan Wuz Here".into(), ..TextItem::default() }).unwrap();
        assert!(strokes.len() > 10);
        // 40 mm text: capitals 0.68 em tall plus the descender-free word.
        assert!(b[3] - b[1] > 25.0 && b[3] - b[1] < 30.0, "{b:?}");
        assert!(find("Arial").unwrap().is_none());
    }
}
