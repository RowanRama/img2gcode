#!/bin/sh
# Install img2gcode from this folder: the CLI goes on PATH and the GUI into
# the application menu.
#
#   ./install.sh               as root: /usr/local; otherwise: ~/.local
#   PREFIX=/opt/x ./install.sh choose the prefix
#   ./install.sh --uninstall   remove what was installed
set -eu

here=$(cd "$(dirname "$0")" && pwd)
if [ -z "${PREFIX:-}" ]; then
    if [ "$(id -u)" = 0 ]; then PREFIX=/usr/local; else PREFIX="$HOME/.local"; fi
fi
bin="$PREFIX/bin/img2gcode"
desktop="$PREFIX/share/applications/img2gcode.desktop"
icon="$PREFIX/share/icons/hicolor/256x256/apps/img2gcode.png"

if [ "${1:-}" = "--uninstall" ]; then
    rm -f "$bin" "$desktop" "$icon"
    echo "Removed img2gcode from $PREFIX"
    exit 0
fi

install -Dm755 "$here/img2gcode" "$bin"
install -Dm644 "$here/img2gcode.png" "$icon"
# Point the menu entry at the installed binary, so it works even when
# $PREFIX/bin is not on the desktop session's PATH.
mkdir -p "$(dirname "$desktop")"
sed "s|^Exec=img2gcode |Exec=$bin |" "$here/img2gcode.desktop" > "$desktop"
chmod 644 "$desktop"
command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database -q "$PREFIX/share/applications" || true
command -v gtk-update-icon-cache >/dev/null 2>&1 && gtk-update-icon-cache -q "$PREFIX/share/icons/hicolor" 2>/dev/null || true

echo "Installed img2gcode to $bin"
case ":$PATH:" in
    *":$PREFIX/bin:"*) echo "Run 'img2gcode --help' for the CLI, or open img2gcode from the application menu." ;;
    *) echo "Add $PREFIX/bin to your PATH to use the CLI, e.g.:"
       echo "  echo 'export PATH=\"$PREFIX/bin:\$PATH\"' >> ~/.profile" ;;
esac
