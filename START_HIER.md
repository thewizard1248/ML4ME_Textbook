# ML4ME: Dein Wegweiser durch den Kurs

Persönliche Orientierung zum offiziellen Lehrbuch von Mark Fuge und dem IDEAL Lab für **Machine Learning for Mechanical Engineering, ETH Zürich**. Erstellt am 17.09.2026 anhand der heruntergeladenen Dateien und des offiziellen Repositorys. Diese Ergänzung stammt nicht vom Kursteam; verbindlich bleiben dessen Kursankündigungen und Aufgabenstellungen.

## In einer Minute starten

1. **Lesen:** Öffne das [Online-Lehrbuch](https://ideal.umd.edu/ML4ME_Textbook/). Links wählst du das Kapitel, innerhalb einer Seite den Abschnitt. Code lässt sich teilweise über „Show Code“ aufklappen. Viele Abbildungen sind dort gespeicherte Ergebnisse; eigene Berechnungen brauchen eine Notebook-Sitzung.
2. **Ausprobieren auf deinem Mac:** Doppelklicke im Finder auf [KURS_STARTEN.command](KURS_STARTEN.command). Es öffnet JupyterLab im Browser. Die Berechnung läuft lokal auf deinem Mac.
3. Öffne links `part1` und dann `reviewing_supervised_linear_models.ipynb`. Wähle bei einer Kernel-Abfrage **ML4ME (Python 3.11)**. Der Kernel ist die Python-Umgebung, die den Code ausführt.
4. Führe Zellen von oben nach unten mit **Shift + Enter** aus. Ändere anschließend einen Parameter und beobachte den Unterschied.
5. Speichere mit **Cmd + S**. Lass das Terminalfenster während der Sitzung offen; zum Beenden dort **Ctrl + C** drücken und gegebenenfalls bestätigen.

Deine GitHub-Kopie: [thewizard1248/ML4ME_Textbook](https://github.com/thewizard1248/ML4ME_Textbook). Das ist ein öffentlicher Fork des [offiziellen Repositorys](https://github.com/IDEALLab/ML4ME_Textbook).

## Was du heruntergeladen hast

Das Repository enthält ein **Lehrbuch mit ausführbaren Experimenten**, Übungsblätter, Daten und die Einrichtung dafür. Es ist kein einzelnes Programm mit einem Startknopf für den ganzen Kurs.

| Datei oder Ordner | Bedeutung | Wann du ihn brauchst |
|---|---|---|
| [`_quarto.yml`](_quarto.yml) | Inhaltsverzeichnis und Einstellungen des veröffentlichten Buchs | Wenn du die verbindliche Buchreihenfolge im Download suchst |
| [`index.qmd`](index.qmd) | Vorwort, Voraussetzungen und Erklärung des Buchkonzepts | Zum Einstieg |
| `part1/` | Grundlagen: lineare Modelle, Zerlegungen, Ableitungen, Verteilungsabstände und Inferenz | Zuerst durcharbeiten |
| `part2/` | Neuronale Netze, generative Modelle, Reinforcement Learning und Transformer | Baut auf Teil 1 auf |
| `part3/` | Active Learning und Semi-Supervised Learning | Lernen mit teuren oder unvollständig beschrifteten Daten |
| `problems/` | Zwei Übungsblätter mit insgesamt fünf Aufgaben-Notebooks | Zum eigenständigen Anwenden |
| `notebooks/` | 13 zusätzliche oder wiederverwendete Unterrichts-Notebooks | Gezielte Vertiefung; zwei davon sind reguläre Kapitel in Teil 1 |
| `appendices/` | Mathematik, SVD und praktische Werkzeuge | Bei Wissenslücken nachschlagen |
| `data/`, `part1/*.npy`, `problems/topo_opt_runs.zip` | Daten für Beispiele und Aufgaben | Werden von bestimmten Notebooks verwendet |
| [`bootstrap_env.py`](bootstrap_env.py) | Offizieller Installer für die Kursumgebung | Einmalige Einrichtung; Wiederholung kann bestehende Umgebung ersetzen |
| [`check_env.py`](check_env.py) | Prüft Bibliotheken und kleine Berechnungen mit Kurscode | Bei der Einrichtung oder technischen Problemen |
| [`pyproject.toml`](pyproject.toml) | Python-Paketliste | Technische Grundlage der Installation |
| `environment.yml`, `environment-gpu.yml` | Conda-Umgebung mit Python 3.11.8 | Offizielles Setup |
| `binder/` | Separate Umgebung für die Online-Ausführung mit Binder | Nur für Binder |
| `.github/workflows/` | Veröffentlichung des Buchs mit Quarto | Für die Pflege des Lehrbuchs |

**Dateitypen:** `.ipynb` ist ein Jupyter-Notebook mit Text, Formeln, Code und gespeicherten Ergebnissen. `.qmd` ist Quarto-Text für Buchseiten. `.py` enthält Python-Hilfsfunktionen oder Programme. `.npy` enthält NumPy-Daten. Zum Lernen musst du Quarto nicht installieren: Die fertig lesbare Buchversion steht bereits online.

Der Download enthält **42 Notebooks**: 6 in Teil 1, 15 in Teil 2, 1 in Teil 3, 13 in `notebooks`, 5 in `problems` und 2 im Anhang. Davon sind 25 direkt in der Buchkonfiguration eingetragen; weitere 5 werden in den Übungsseiten eingebettet. Die übrigen 12 sind zusätzliche Dateien außerhalb des direkten Inhaltsverzeichnisses. Deshalb ist alphabetisches Durchklicken im Dateibrowser keine gute Kursreihenfolge.

## Der rote Faden

**Teil 1 beantwortet: Wie baue, trainiere und beurteile ich ein Modell?** Du lernst zuerst Vorhersage, Fehlermaße und Regularisierung. Cross-Validation prüft, ob ein Modell auch mit neuen Daten funktioniert. SGD und automatische Differentiation erklären, wie Parameter gelernt werden. PCA und verwandte Zerlegungen reduzieren hochdimensionale Geometrien. Verteilungsabstände und Inferenz schaffen die Grundlage dafür, später ganze Datenverteilungen und Unsicherheit zu modellieren.

**Teil 2 beantwortet: Wie lassen sich neue Daten oder technische Entwürfe erzeugen?** Der Schwerpunkt liegt deutlich auf generativen Modellen. GANs, VAEs, Normalizing Flows, Diffusion und Flow Matching verfolgen ein gemeinsames Ziel mit unterschiedlichen Trainingsprinzipien. Wiederkehrende kleine Datensätze, etwa ein Ring aus Gauß-Verteilungen, machen die Verfahren vergleichbar. Reinforcement Learning ergänzt das Lernen von Entscheidungen; Transformer zeigen den Umgang mit Mengen und Punktwolken.

**Teil 3 beantwortet: Welche Daten sollte ich als Nächstes beschaffen?** Bei teuren Simulationen oder Experimenten zählt der Informationsgewinn jeder neuen Messung. Active Learning wählt gezielt neue Datenpunkte; Semi-Supervised Learning nutzt zusätzlich Daten ohne Labels.

Mechanik-Bezüge sind beispielsweise Tragflügelgeometrien, Materialparameter aus Spannungs-Dehnungs-Daten, Topologieoptimierung und dynamische Systeme. Das Buch setzt Grundlagen aus Stochastik, linearer Algebra und Differentialrechnung voraus. Bei Lücken sind die Anhänge als Nachschlagewerk gedacht.

## Lesereihenfolge mit direkten Dateien

Die folgenden Nummern sind die Kapitelnummern der aktuellen Buchstruktur. Die Dateinamen sind teils älter als die angezeigten Kapiteltitel.

| Kapitel | Thema und Zweck | Notebook |
|---|---|---|
| 2 | Lineare Modelle, Overfitting, Regularisierung und Loss-Funktionen | [reviewing_supervised_linear_models](part1/reviewing_supervised_linear_models.ipynb) |
| 3 | Modelle fair bewerten; Cross-Validation und Hyperparameter | [cross_validation_linear_regression](notebooks/cross_validation_linear_regression.ipynb) |
| 4 | Gradient Descent und SGD verstehen | [supervised_linear_models](notebooks/supervised_linear_models.ipynb) |
| 5 | PCA, Sparse PCA, Dictionary Learning und NMF | [linear_decompositions](part1/linear_decompositions.ipynb) |
| 6 | Automatische Ableitungen und PyTorch Autograd | [taking_derivatives](part1/taking_derivatives.ipynb) |
| 7 | KL, Jensen–Shannon, MMD und Optimal Transport | [distribution_distance](part1/distribution_distance.ipynb) |
| 8 | Bayes, MCMC und Variational Inference | [introduction_to_inference](part1/introduction_to_inference.ipynb) |
| 9 | Probabilistische Modelle praktisch mit Pyro | [introduction_to_probabilistic_programming](part1/introduction_to_probabilistic_programming.ipynb) |
| 10 | Neuronale Netze und Autoencoder auffrischen | [review_neural_networks](part2/review_neural_networks.ipynb) |
| 11 | GANs: Generator und Diskriminator | [intro_to_GANS](part2/gen_models/intro_to_GANS.ipynb) |
| 12 | GAN-Training verstehen und Fehler diagnostizieren | [GAN_pitfalls](part2/gen_models/GAN_pitfalls.ipynb) |
| 13 | Optimal Transport als Trainingsprinzip | [OT](part2/gen_models/OT.ipynb) |
| 14 | Variational Autoencoders | [VAEs](part2/gen_models/VAEs.ipynb) |
| 15 | Invertierbare Transformationen und Dichten | [normalizing_flows](part2/gen_models/normalizing_flows.ipynb) |
| 16 | Kontinuierliche Flows und neuronale ODEs | [continuous_flows](part2/gen_models/continuous_flows.ipynb) |
| 17 | Gradienten der Log-Dichte lernen | [score_matching](part2/gen_models/score_matching.ipynb) |
| 18 | Rauschen schrittweise rückgängig machen | [diffusion_models](part2/gen_models/diffusion_models.ipynb) |
| 19 | Ein Transport-Geschwindigkeitsfeld lernen | [flow_matching](part2/gen_models/flow_matching.ipynb) |
| 20 | Generative Modelle im komprimierten Raum | [latent_generative_models](part2/gen_models/latent_generative_models.ipynb) |
| 21 | DQN, Policy Gradients, Actor-Critic und PPO | [introduction_to_reinforcement_learning](part2/introduction_to_reinforcement_learning.ipynb) |
| 22 | Attention und Transformer für Formen und Punktwolken | [transformers](part2/gen_models/transformers.ipynb) |
| 23 | Active und Semi-Supervised Learning | [active_and_semisupervised_learning](part3/active_and_semisupervised_learning.ipynb) |

Danach folgen die Übungsblätter, die [Housing-Visualisierungsübung](notebooks/california_housing_visualization.ipynb) und die Anhänge. Nützliche Nachschlagestellen sind [SVD](appendices/review_of_singular_value_decomposition.ipynb), [Mathematik und Computing](appendices/review_of_math_and_computing_foundations.ipynb) und [Debugging/Tooling](appendices/helpful_tooling.qmd).

### Welche Übungen gehören wohin?

| Aufgabe | Inhalt | Sinnvolle Vorbereitung |
|---|---|---|
| [PS1 Teil 1](problems/ps1_part1.ipynb) | Lineare Modelle und Validierung | Kapitel 2–3 |
| [PS1 Teil 2](problems/ps1_part2.ipynb) | PCA und Topologieoptimierung | Kapitel 5 |
| [PS1 Teil 3](problems/ps1_part3.ipynb) | Bilineares Empfehlungssystem mit SGD | Kapitel 4–5 |
| [PS2 Teil 1](problems/ps2_part1_autodiff.ipynb) | Ableitungen, Tankoptimierung, springender Ball | Kapitel 6 |
| [PS2 Teil 2](problems/ps2_part2.ipynb) | GANs für Tragflügel verbessern und vergleichen | Kapitel 11–12 |

`ps1.qmd` und `ps2.qmd` bündeln diese Notebooks zur Buchansicht. **Zum Bearbeiten öffnest du die `.ipynb`-Dateien.** Diese Zuordnung ist eine Lernhilfe, keine Aussage über Abgabefristen.

## So arbeitest du im Alltag

1. Lies einen Abschnitt im Online-Buch und formuliere in einem Satz, was er erklären soll.
2. Öffne das passende Notebook und führe die bisherigen Zellen in Reihenfolge aus. Eine Zelle kann Variablen oder Funktionen aus früheren Zellen benötigen.
3. Bei einem Experiment: erst eine Vorhersage machen, dann genau einen Parameter ändern, dann das Ergebnis erklären.
4. Für eigene Versuche kannst du im selben Ordner eine Kopie anlegen, etwa `taking_derivatives_meine_notizen.ipynb`. So bleiben relative Datenpfade und Hilfsmodule erreichbar.
5. Vor einer Abgabe: Kernel neu starten und alle Zellen von oben ausführen. So entdeckst du versehentlich verwendete Variablen aus alten Versuchen. Bei langen Trainings zunächst einzelne Abschnitte ausprobieren.

**Speichern und GitHub sind zwei Schritte:** Cmd + S speichert lokal. Erst ein Git-Commit und Push sichern ausgewählte Änderungen auf GitHub. Der Fork ist öffentlich; halte dich beim Hochladen eigener Lösungen an die Vorgaben eures Kurses.

## Einrichtung auf dem Mac

Die Kursumgebung heißt `ml4me-student`. Sie nutzt Python **3.11.8** und die im offiziellen Installer vorgegebenen Pakete **torch 2.7.1**, **torchvision 0.22.1**, **torchaudio 2.7.1**. Miniforge liegt unter `~/miniforge3`. Das vorhandene Python 3.14 bleibt davon getrennt.

Der Starter öffnet JupyterLab mit der Kursumgebung direkt. Dafür ist keine dauerhafte Änderung deiner Terminal-Konfiguration nötig. Wenn macOS beim Doppelklick das Öffnen verweigert, öffne ein Terminal im Kursordner und führe `bash KURS_STARTEN.command` aus.

### Manuell starten oder prüfen

Öffne ein Terminal im Kursordner und führe aus:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate ml4me-student
python check_env.py
jupyter lab
```

Den Buch-Ordner kannst du auch in **VS Code** öffnen, falls du später einen Editor mit Git-Integration bevorzugst. Installiere dort die Microsoft-Erweiterungen „Python“ und „Jupyter“, öffne ein Notebook und wähle oben rechts **ML4ME (Python 3.11)** als Kernel. VS Code ist für die Browser-Variante nicht erforderlich.

### Neuinstallation auf einem anderen Apple-Silicon-Mac

Zuerst [Miniforge für macOS arm64](https://github.com/conda-forge/miniforge) installieren und den Fork klonen oder herunterladen. Dann im Kursordner:

```bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda env create -f environment.yml
conda activate ml4me-student
python -m pip install "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1"
python -m pip install . "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" jupyterlab ipykernel requests tqdm
python -m ipykernel install --user --name ml4me-student --display-name "ML4ME (Python 3.11)"
python check_env.py
```

Diese Schritte sind für eine **neue** Umgebung gedacht. Wenn `ml4me-student` bereits existiert, aktiviere und prüfe sie zuerst. Die offizielle Paketliste enthält überwiegend Mindestversionen; eine spätere Installation muss deshalb nicht dieselben Versionen ergeben wie heute. [SETUP_STATUS.md](SETUP_STATUS.md) dokumentiert die hier tatsächlich geprüfte Einrichtung.

Für dieselben Paketversionen auf einem weiteren Apple-Silicon-Mac kannst du nach dem Anlegen und Aktivieren der Python-3.11.8-Umgebung die beiden `pip install`-Zeilen durch `python -m pip install -r requirements-mac-tested.txt` ersetzen. Die [Paketliste](requirements-mac-tested.txt) ist eine Momentaufnahme dieser Installation und keine allgemeine Windows- oder Linux-GPU-Konfiguration.

Auf Apple Silicon heißt die GPU-Schnittstelle **MPS**, nicht CUDA. Einige Kursnotebooks wählen MPS automatisch, andere verwenden auf dem Mac die CPU. Eine Meldung `CUDA unavailable` allein bedeutet daher keinen kaputten Mac-Setup. Stelle nicht pauschal alle Notebooks auf MPS um; unterstützte Operationen und Datentypen unterscheiden sich.

## Gefundene Stolperstellen im offiziellen Material

| Beobachtung | Bedeutung und Vorgehen |
|---|---|
| `hb.csv` fehlt lokal und der in PS1 Teil 2 eingetragene offizielle Download liefert 404 | Der erste PCA-Aufgabenteil benötigt diese Datei vom Kursteam. Das ist kein Installationsfehler. |
| `ratings.csv` und `missing.csv` fehlen lokal und unter den geprüften Pfaden im offiziellen Repository | PS1 Teil 3 kann seinen Datensatz so nicht laden. Originaldateien vom Kursteam beschaffen und in `problems/` ablegen. |
| `engibench_checkpoints.zip` fehlt | Das zusätzliche `EngiBench_latent_gen_models.ipynb` benötigt vortrainierte Modelle. Es ist nicht das reguläre Kapitel `latent_generative_models.ipynb`. |
| Einige Notebooks laden Daten aus dem Internet, obwohl ähnliche Daten lokal liegen | Die erste Ausführung kann eine Internetverbindung benötigen, etwa bei Tragflügeln oder California Housing. |
| `bootstrap_env.py` entfernt eine vorhandene Umgebung nach fehlgeschlagener Neuanlage | Nicht als täglichen Startknopf verwenden. Der mitgelieferte Starter startet nur Jupyter. |
| `check_env.py` kontrolliert CUDA, aber nicht MPS | Seine GPU-Warnung ist auf einem Apple-Silicon-Mac allein kein Fehler. MPS separat prüfen. |
| Der Prüfer bezeichnet einige inzwischen deklarierte Pakete weiterhin als „optional“ und erwähnt `geomstats`, obwohl das aktuelle Verteilungsabstands-Notebook es nicht importiert | Warnungen immer auf das tatsächlich verwendete Notebook beziehen. |
| `.gitignore` ignoriert pauschal `notebooks/` und CSV-Dateien | Bereits versionierte Kursnotebooks bleiben erfasst. Neue eigene Notebooks in diesem Ordner und neue Datensätze erscheinen möglicherweise nicht automatisch bei Git. Eine gewünschte Datei gezielt hinzufügen, gegebenenfalls `git add -f pfad/zur/datei.ipynb`. |
| Übersichtsseiten erwähnen teilweise weitere Themen, die nicht als Kapitel eingebunden sind | Die tatsächliche Reihenfolge aus `_quarto.yml` verwenden. Nicht jede `.qmd`-Datei gehört zur veröffentlichten Fassung. |

Bei `NameError` zuerst frühere Zellen ausführen. Bei `ModuleNotFoundError` zuerst den Kernel prüfen. Bei fehlenden lokalen Dateien im Notebook `import os; print(os.getcwd())` ausführen: Das Arbeitsverzeichnis sollte zum Notebook und seinen relativen Pfaden passen. Notebooks aus `part2/gen_models/` am besten in diesem Ordner belassen, da sie dortige `.py`-Hilfsmodule importieren.

## GitHub und Updates

`origin` zeigt auf deinen Fork, `upstream` auf das offizielle Kursrepository. Der lokale Ordner wurde mit der Git-Historie verbunden, sodass Updates nachvollziehbar bleiben.

Für den Vergleich mit neuen Kursunterlagen:

```bash
git fetch upstream
git log --oneline HEAD..upstream/main
```

Wenn keine ungesicherten Änderungen vorliegen (`git status` prüfen), kannst du die Updates übernehmen:

```bash
git merge upstream/main
git push origin main
```

Bei Konflikten hält Git an: Dann die betroffenen Änderungen gemeinsam auflösen, statt lokale Arbeit zu überschreiben. Eigene ausgewählte Änderungen sicherst du mit `git add DATEI`, `git commit -m "Beschreibung"` und `git push origin main`.

Die ursprüngliche Autorenschaft und der im Lehrbuch genannte Lizenzhinweis **CC BY-NC-SA 4.0** bleiben erhalten. Dieser Wegweiser und der Starter sind persönliche Ergänzungen zum Kursmaterial.

## Mein Vorschlag für deine erste Sitzung

Starte mit Kapitel 2, führe das erste Regressionsbeispiel aus und variiere Modellkomplexität und Regularisierung. Gehe danach zu Kapitel 3 und prüfe, ob die scheinbare Verbesserung auch auf Validierungsdaten hält. Wenn du diesen Unterschied erklären kannst, hast du einen sehr nützlichen Einstieg für den restlichen Kurs.
