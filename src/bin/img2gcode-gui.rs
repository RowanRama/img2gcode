//! The GUI on its own, built as a Windows GUI program so launching it from a
//! shortcut opens no console window. `img2gcode` is the CLI.
#![windows_subsystem = "windows"]

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    // A file passed by "Open with" or a drag onto the icon.
    let file = std::env::args_os().nth(1).map(PathBuf::from);
    let is_gcode = file
        .as_ref()
        .and_then(|p| p.extension())
        .and_then(|e| e.to_str())
        .is_some_and(|e| ["gcode", "gco", "nc"].iter().any(|g| e.eq_ignore_ascii_case(g)));
    let (input, gcode_file) = if is_gcode { (None, file) } else { (file, None) };
    match img2gcode::gui::run(img2gcode::Config::default(), input, gcode_file) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // There is no console to print to.
            rfd::MessageDialog::new()
                .set_title("img2gcode")
                .set_description(e)
                .set_level(rfd::MessageLevel::Error)
                .show();
            ExitCode::FAILURE
        }
    }
}
