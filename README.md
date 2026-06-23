# Bierkasternerkennung

Computer-Vision-Projekt zur **Lokalisierung eines Bierkastens** im Kamerabild (Arbeitspaket *Kastendetektion*): normierte Bounding Box **(x, y, w, h)** und **vier Eckpunkte** für das nachgelagerte Grid Mapping / Warp Perspective.

## Frameworks (nicht neu erfinden)

| Aufgabe | Bibliothek |
|--------|------------|
| Objektdetektion, Training, Export `best.pt` | **[Ultralytics YOLOv8](https://docs.ultralytics.com/)** (`pip install ultralytics`) |
| Kamera, Kanten, Konturen, Zeichnen | **OpenCV** (`cv2`) |

YOLOv8 ist der übliche Standard für solche Bounding-Box-Aufgaben; das Arbeitspaket verlangt explizit Ultralytics und Fine-Tuning auf die Klasse `bierkasten`.

## Abgleich: `Projektplan_Bierkasten_Hybrid.docx`

| Baustein im Plan | Stand im Repo |
|------------------|---------------|
| Kamera-Stream (`cv2.VideoCapture`) | Umgesetzt (`scripts/camera_demo.py`) |
| Kastenlokalisierung YOLOv8 + eigene Klasse „Bierkasten“ | Umgesetzt (`detect_crate`, `kasten.yaml` / Label „bierkasten“ = ID 0) |
| Fallback Canny + Konturen | Umgesetzt |
| Eckpunkte + Orientierung | Ecken ja; **Orientierung** als `orientation_deg` beim **Konturpfad** (OpenCV `minAreaRect`); bei **YOLO** aktuell `None` (Achsen-BBox) |
| Warp Perspective → normierte Draufsicht | Umgesetzt (`kastendetektion/warp_grid.py`: `warp_crate_top_down`) |
| Grid Mapping 4×5, 20 Slot-Mittelpunkte | Umgesetzt (`grid_slot_centers`) |
| Stufe 1: Slot voll/leer (Helligkeit/Kanten/Hough + CNN) | Umgesetzt (`slot_classifier.py`, klassisch + KI) |
| Stufe 2: Kronkorken / voll-leer (HSV/Specular + CNN) | Umgesetzt (`cap_classifier.py`, klassisch + KI) |
| Overlay grün/gelb/rot + Gesamtstatistik | Umgesetzt (`overlay.py`, `scripts/pipeline_demo.py`) |
| Frame-Stabilisierung | Umgesetzt (`stabilize.py`, Mehrheitsentscheid je Slot) |
| ROI-Datensatz + CNN-Training (Stufe 1/2) | Tooling umgesetzt (`extract_slot_dataset.py`, `train_slot_cnn.py`, `train_cap_cnn.py`); Training mit eigenen ROIs offen |

Die vier Ecken aus dem **YOLO-Rechteck** sind eine **Näherung** für Warp (nicht die exakte Physik-Oberfläche); für mehr Genauigkeit später echte Oberflächen-Ecken annotieren oder Pose/OBB erweitern.

## Projektstruktur

- `kasten.yaml` — Dataset-Konfiguration für `yolo train`
- `kastendetektion/detector.py` — **`detect_crate(frame)`** (YOLO primär, Canny/Kontur als Fallback)
- `kastendetektion/warp_grid.py` — **Warp Perspective** + **4×5 Slot-Mittelpunkte** (+ ROI-Helfer für spätere CNNs)
- `scripts/prepare_dataset.py` — flache `images/` + `labels/` → `train/` / `val/`
- `scripts/train_yolo.py` — Fine-Tuning (`epochs` Standard 50)
- `scripts/verify_labels.py` — prüft YOLO-Labels (Klasse 0, Werte in [0, 1])
- `scripts/auto_label.py` — **YOLO-Startlabels** per CV (Rot + Kontur-Fallback)
- `scripts/label_ui.py` — **Streamlit-Oberfläche**: Bilder labeln, Klassen als Dropdown (siehe unten)
- `labeling/classes.txt` — Klassennamen für die UI (eine Zeile = eine Klasse, ID 0, 1, …)
- `scripts/camera_demo.py` — Live-Stream mit Overlay
- `scripts/classical_demo.py` — nur klassischer Pfad (Canny) auf einem Testbild

Trainingsdaten und `runs/` sind per `.gitignore` ausgeschlossen; **`best.pt`** laut Vorgabe ins Team-Laufwerk legen, nicht ins Repo.

## Schnittstelle fürs nächste Modul

```python
from kastendetektion import detect_crate, warp_crate_top_down, grid_slot_centers

result = detect_crate(frame_bgr, log_corners=True)  # frame_bgr: numpy BGR
if result:
    x, y, w, h = result.x, result.y, result.w, result.h
    corners = result.corners   # shape (4, 2), float32 — TL, TR, BR, BL
    conf = result.confidence   # 0..1
    source = result.source     # "yolo" oder "contour"
    angle = result.orientation_deg  # nur Konturpfad: OpenCV-Winkel, sonst None

    warped, H = warp_crate_top_down(frame_bgr, corners, out_width=500, out_height=400)
    centers = grid_slot_centers(500, 400, rows=4, cols=5)  # 20 Punkte, zeilenweise
```

**Erkannte Gewichte (optional):** Umgebungsvariable `KASTEN_YOLO_WEIGHTS` auf `best.pt` setzen, oder `weights_path="..."` übergeben. Suchreihenfolge: Argument → `KASTEN_YOLO_WEIGHTS` → `runs/detect/kasten/weights/best.pt` → `runs/detect/train/weights/best.pt`. Ohne eigene Gewichte wird automatisch der **Kontur-Fallback** genutzt.

## Füllzustand-Pipeline (Stufe 1 + Stufe 2)

Nach der Kastendetektion bestimmt die Pipeline pro Slot den Zustand und erzeugt ein farbcodiertes Overlay:

- **Stufe 1 – Slot belegt/leer** (`kastendetektion/slot_classifier.py`): Kantendichte + Hough-Kreis (klassisch) **oder** CNN (KI).
- **Stufe 2 – Kronkorken voll/leer** (`kastendetektion/cap_classifier.py`): HSV/Specular-Heuristik (klassisch) **oder** CNN (KI), nur für belegte Slots.
- **Zustände** (`states.py`): `FULL` = grün (Flasche + Korken), `EMPTY` = gelb (Flasche ohne Korken), `MISSING` = rot (leer).

Beide Stufen kennen `method = "classical" | "ml" | "auto"` (`auto`: KI, wenn Gewichte vorhanden, sonst klassisch). KI-Gewichte über `KASTEN_SLOT_WEIGHTS` / `KASTEN_CAP_WEIGHTS` oder Argument.

```python
from kastendetektion import CratePipeline, draw_pipeline_overlay

pipe = CratePipeline(slot_method="auto", cap_method="auto")
analysis = pipe.analyze(frame_bgr)
if analysis:
    vis = draw_pipeline_overlay(frame_bgr, analysis)
    print(analysis.occupied_count, "belegt,", analysis.full_count, "voll")
```

### Pipeline-Demo

```bash
# Einzelbild (z. B. synthetischer Testkasten), klassischer Pfad, Ergebnis speichern
python scripts/pipeline_demo.py --image demo_output/synthetic_crate.png \
    --slot-method classical --cap-method classical --save demo_output/05_pipeline.png

# Live-Kamera mit Frame-Stabilisierung über 10 Frames
python scripts/pipeline_demo.py --camera 0 --stabilize 10
```

### Bedien-UI (Streamlit)

Komfortable Oberfläche zum Bedienen der Pipeline (Bild-Upload oder Kamera, Methode je Stufe umschaltbar, Overlay + Statistik + Slot-Tabelle):

```bash
streamlit run scripts/app_ui.py
```

Öffnet sich kein Fenster automatisch, im Browser **http://localhost:8501** aufrufen. In der Seitenleiste:

- "Nur Kontur statt YOLO" abwählen → Detektion läuft über YOLO (`best.pt` per Feld oder `KASTEN_YOLO_WEIGHTS`).
- "Stufe 1/2" auf `classical` = klassische Methode, auf `ml` = CNN, `auto` = CNN wenn Gewichte vorhanden, sonst klassisch.

### KI-Wege trainieren (Stufe 1/2)

1. Slot-ROIs aus gelabelten Kastenbildern extrahieren:

```bash
python scripts/extract_slot_dataset.py --images data/kasten_dataset/images
```

2. ROIs aus `data/slot_dataset/unsorted/` manuell in `empty/`, `bottle_empty/`, `bottle_full/` einsortieren.
3. Klassifikatoren trainieren (kleines CNN oder MobileNetV2):

```bash
python scripts/train_slot_cnn.py --epochs 30          # -> models/slot_cnn.pt
python scripts/train_cap_cnn.py  --epochs 30          # -> models/cap_cnn.pt
```

4. Inferenz mit KI-Wegen:

```bash
export KASTEN_SLOT_WEIGHTS=models/slot_cnn.pt
export KASTEN_CAP_WEIGHTS=models/cap_cnn.pt
python scripts/pipeline_demo.py --camera 0 --slot-method ml --cap-method ml
```

### Tests

```bash
pip install pytest   # Dev-Abhängigkeit
python -m pytest tests/ -q
```

## Daten von Dropbox

Trainingsbilder aus dem Kursordner:  
[Dropbox — Trainingsbilder](https://www.dropbox.com/scl/fo/0d5055swt00lm3oemtv30/AA5mTgKvz6-RsGlngpAuu_Y?rlkey=usqutwwa68w591k9ch2frowjw&e=1&st=m6kvlxcf&dl=0)

1. Ordner **`data/kasten_dataset/images/`** nutzen (liegt im Projekt; dort liegen auch `HINWEIS.txt` und `.gitkeep` — **eigene Fotos werden von Git ignoriert**).
2. **Start-Labels automatisch erzeugen** (Rot-Kasten-Detektion, bei Bedarf Kontur-Fallback):

```bash
python scripts/auto_label.py
# optional: Overlay prüfen
python scripts/auto_label.py --preview-dir data/kasten_dataset/_auto_preview
```

Vorhandene `.txt` werden standardmäßig **nicht** überschrieben (`--skip-existing`). Alle neu erzeugen: `python scripts/auto_label.py --overwrite`.

3. **Stichprobe / Korrektur** — fehlgeschlagene oder ungenaue Boxen in der Label-UI oder mit LabelImg nachziehen (siehe unten).
4. Labels prüfen:

```bash
python scripts/verify_labels.py
```

`verify_labels.py` erwartet pro Bild eine gültige `.txt`, sobald ein Kasten sichtbar ist. Bilder ohne erkannte Box nach Schritt 2 manuell labeln oder aus dem Ordner entfernen.

5. Split ausführen (im Repo-Root):

```bash
python scripts/prepare_dataset.py
```

6. Training:

```bash
python scripts/train_yolo.py --epochs 50
```

CLI analog zum Arbeitspaket:

```bash
yolo train model=yolov8n.pt data=kasten.yaml epochs=50
```

7. Inferenz mit Gewichten z. B.:

```bash
set KASTEN_YOLO_WEIGHTS=runs\detect\kasten\weights\best.pt
python scripts/camera_demo.py
```

### Lokale Label-UI (Klassen-Dropdown)

Im Projektroot:

```bash
pip install -r requirements.txt
streamlit run scripts/label_ui.py
```

Öffnet sich kein Fenster: im Browser **http://127.0.0.1:8501** (oder `http://localhost:8501`) aufrufen.

Mit **`--server.headless true`** öffnet Streamlit **keinen** Browser automatisch — dann immer den Link manuell öffnen.

Wenn die **Canvas ohne Hintergrundbild** fehlschlägt: neuere Streamlit-Versionen haben `image_to_url` umgebaut — `scripts/label_ui.py` patcht das automatisch für `streamlit-drawable-canvas`.

- Bildordner standardmäßig **`data/kasten_dataset/images/`** (dorthin Kursfotos legen).  
- Klassen bearbeiten in der Seitenleiste oder direkt in **`labeling/classes.txt`** — jede Zeile erscheint als Dropdown-Eintrag (**erste Zeile = Klasse 0**).  
- Pro gezeichnete Box ein Dropdown; Speichern schreibt **YOLO-Format** nach **`data/kasten_dataset/labels/`**.

**Hinweis Arbeitspaket:** Für das vorgegebene Training ist aktuell nur **`bierkasten`** vorgesehen (`kasten.yaml`: eine Klasse). Wenn du **mehrere Klassen** labelst, musst du `kasten.yaml` (`nc`, `names`) und ggf. `scripts/verify_labels.py` entsprechend anpassen.

### LabelImg Schritt für Schritt

**Installation (eine Variante reicht):**

```bash
pip install labelImg
labelImg
```

Oder das fertige Release von **[labelImg auf GitHub](https://github.com/HumanSignal/labelImg/releases)** nutzen.

**Einstellungen vor dem ersten Rahmen:**

1. **Open Dir** → Ordner `data/kasten_dataset/images/` (nur Bilder, keine Labels mischen).
2. **Change Save Dir** → `data/kasten_dataset/labels/` (Ultralytics erwartet parallel zu `images/` einen Ordner `labels/`).
3. Links unten das Format auf **YOLO** stellen (nicht PascalVOC/XML).
4. Beim ersten Kasten die Klasse **`bierkasten`** anlegen — später immer dieselbe Klasse nutzen (ein Klassenname → bei uns Index **0** in den `.txt`-Dateien).

**Annotieren:**

- **`w`** — nächstes Bild  
- **`a`** — vorheriges Bild  
- **`d`** — Box zeichnen (RectBox), dann Rechteck um den **ganzen sichtbaren Kasten** ziehen (leicht luftig zu den Kanten ist okay).  
- **`Ctrl+S`** / **Save** — speichert `bildname.txt` neben dem Bildnamen in den Label-Ordner.  
- Pro Bild **mindestens eine Box**, wenn ein Kasten sichtbar ist. Bilder ohne Kasten brauchen **keine** `.txt` (oder leere Datei — für Training ohne Objekt ist das eine andere Konvention; für dieses Projekt reicht: nur Bilder mit Kasten labeln).

**Qualität laut Arbeitspaket:** verschiedene **Winkel**, **Rotation**, **Licht** — das verbessert später Warp Perspective und Robustheit.

**Nach dem Labeln:** `python scripts/verify_labels.py` ausführen. Meldungen zu falscher Klassen-ID oder Koordinaten außerhalb `[0,1]` vor dem Training beheben.

### Training und Hyperparameter

`scripts/train_yolo.py` wählt sinnvolle **Standards**:

| Situation | Standard im Skript |
|-----------|---------------------|
| GPU (CUDA) | `device=0`, **`batch=-1`** (AutoBatch von Ultralytics) |
| nur CPU | `device=cpu`, **`batch=4`** (RAM-schonend; bei genug RAM `--batch 8` testen) |
| Windows | **`workers=0`** (weniger DataLoader-Probleme); auf Linux optional `--workers 8` |
| Early Stopping | **`patience=25`** (anpassbar, wenn das Training zu früh stoppt oder zu lange läuft) |

Weitere sinnvolle Knöpfe:

```bash
# Schnelleres Training / weniger VRAM (oft noch okay bei großen Objekten im Bild)
python scripts/train_yolo.py --imgsz 416 --batch 8

# Größeres Modell (genauer, langsamer)
python scripts/train_yolo.py --model yolov8s.pt

# Wenige Bilder: Cache kann die Epoch-Zeit verkürzen (mehr RAM)
python scripts/train_yolo.py --cache
```

**mAP-Ziel (> 0,8):** hängt stark von **einheitlicher Labelqualität** und **Varianz** der Bilder ab. Bei kleinen Datensätzen lieber mehr Epochen oder weniger starkes Early Stopping (`--patience 50`), nach Metriken unter `runs/detect/` beurteilen.

## Kurzablauf Arbeitspaket

| Schritt | Befehl / Hinweis |
|--------|-------------------|
| Umgebung | `pip install -r requirements.txt` |
| Kamera | `python scripts/camera_demo.py` |
| Klassisch Testbild | `python scripts/classical_demo.py pfad/zum/bild.jpg` |
| Nur Fallback live | `python scripts/camera_demo.py --contour-only` |
| Eckpunkte loggen | `python scripts/camera_demo.py --log-corners` |
| Auto-Labels | `python scripts/auto_label.py` |
| Labels prüfen | `python scripts/verify_labels.py` |
| Label-Web-UI | `streamlit run scripts/label_ui.py` |
| Füllzustand-Pipeline | `python scripts/pipeline_demo.py --image demo_output/synthetic_crate.png` |
| Bedien-UI (Pipeline) | `streamlit run scripts/app_ui.py` |

Voraussetzung für sinnvolles YOLO: zuerst Daten labeln und trainieren; bis dahin liefert die Pipeline den **Canny-Fallback** für erste Demos.

## Contributors
@RobinMueller94
@miascharpf
@DaKrieger-stack
