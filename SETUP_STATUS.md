# Geprüfte Einrichtung – 17.09.2026

Persönliche Ergänzung zum offiziellen ML4ME-Kursmaterial.

## Installiert

- Miniforge 26.7.2-0 für macOS arm64, unter `~/miniforge3`.
- Separate Conda-Umgebung `ml4me-student`, Python 3.11.8.
- Offizielle Kursabhängigkeiten mit torch 2.7.1, torchvision 0.22.1 und torchaudio 2.7.1.
- JupyterLab 4.6.3 und Notebook-Kernel **ML4ME (Python 3.11)**.
- Starter [KURS_STARTEN.command](KURS_STARTEN.command) und [deutscher Wegweiser](START_HIER.md).
- Vollständige Versionsliste: [requirements-mac-tested.txt](requirements-mac-tested.txt).

Die vorhandene systemweite Python-3.14-Installation wurde nicht verändert. Miniforge wurde ohne automatische Shell-Initialisierung installiert. Der Starter aktiviert die Kursumgebung für seine Sitzung.

## Verifiziert

- Miniforge-Installer: SHA-256 mit dem offiziellen Release abgeglichen.
- Paketabhängigkeiten: `python -m pip check` ohne Fehler.
- Offizieller Prüfer: `python check_env.py` mit Exit-Code 0; alle Pflichtprüfungen bestanden.
- Kurscode: `CGANGenerator` erzeugt einen Tensor der Form `(2, 1, 64, 64)`; die Ring-Gauß-Datengenerierung funktioniert.
- Apple-GPU: MPS verfügbar; Matrixmultiplikation gegen das erwartete Ergebnis geprüft.
- `autodiff_pytorch_simple_function.ipynb`: alle 9 Codezellen in einem frischen Kurskernel ausgeführt, ohne Fehler.
- Einstieg in `reviewing_supervised_linear_models.ipynb`: alle 7 Codezellen des ersten Abschnitts bis einschließlich der Ridge-Regularisierungsbeispiele in einem frischen Kernel ausgeführt, ohne Fehler.
- Vollständiger Test von `reviewing_supervised_linear_models.ipynb`: beim umfangreichen Optimierungsvergleich mit `scipy.optimize.fmin` (14 Regularisierungen × 12 Normen) nach dem gesetzten Limit von 180 Sekunden für diese Zelle abgebrochen. Die vollständige Ausführung dieses Kapitels ist damit nicht bestätigt; das Notebook wurde nicht verändert.
- Starter über Bash ausgeführt; JupyterLab läuft im richtigen Kursordner. Weboberfläche und Zugriff auf das erste Kapitel liefern HTTP 200.
- Git: alle 42 ursprünglichen Notebooks werden versioniert; der Download stimmte vor den persönlichen Ergänzungen vollständig mit dem aktuellen `main` überein.

## Einordnung der drei Warnungen des offiziellen Prüfers

1. **Kein CUDA:** Auf diesem Apple-Silicon-Mac erwartbar. MPS wurde separat erfolgreich geprüft.
2. **`wandb` fehlt:** Wird im zusätzlichen EngiBench-Checkpoint-Notebook importiert. Dieses benötigt außerdem die im Repository fehlenden Modell-Checkpoints. Für den geprüften Einstieg nicht erforderlich.
3. **`geomstats` fehlt:** Der Prüfer nennt es für `distribution_distance.ipynb`; im untersuchten Notebook wird es nicht importiert.

Nicht sämtliche 42 Notebooks oder langen Trainingsläufe wurden vollständig ausgeführt. Der erfolgreiche Umgebungscheck ersetzt keine vollständige fachliche Prüfung aller Kursbeispiele. Die in [START_HIER.md](START_HIER.md) aufgeführten fehlenden Übungsdatensätze und Checkpoints bleiben separate Einschränkungen des bereitgestellten Materials.

## GitHub

- Eigener öffentlicher Fork: [thewizard1248/ML4ME_Textbook](https://github.com/thewizard1248/ML4ME_Textbook).
- Offizielle Quelle: [IDEALLab/ML4ME_Textbook](https://github.com/IDEALLab/ML4ME_Textbook).
- Geprüfter offizieller Stand: `ae7600d4bc52e31e95772b7934024afd9205bd2b`.
- Lokale Verbindungen: `origin` = eigener Fork; `upstream` = offizieller Kurs.

Die persönlichen Ergänzungen ändern keine Aufgabenstellungen und enthalten keine neu erstellten Übungslösungen.
