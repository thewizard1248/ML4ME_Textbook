#!/bin/bash
set -euo pipefail

KURS_ORDNER="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$KURS_ORDNER"

if [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Miniforge wurde nicht gefunden. Siehe START_HIER.md zur Einrichtung."
    read -r -p "Enter zum Schliessen. "
    exit 1
fi

if ! conda activate ml4me-student; then
    echo "Die Kursumgebung ml4me-student fehlt. Siehe START_HIER.md."
    read -r -p "Enter zum Schliessen. "
    exit 1
fi

echo "ML4ME startet im Browser. Dieses Fenster waehrend der Sitzung offen lassen."
echo "Zum Beenden hier Ctrl+C druecken."
exec python -m jupyterlab --ServerApp.root_dir="$KURS_ORDNER" --ServerApp.ip=127.0.0.1
