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
| Stufe 1: Slot voll/leer (Hough, Helligkeit, CNN/SVM) | **Noch nicht** |
| Stufe 2: Kronkorken / voll-leer (HSV, CNN) | **Noch nicht** |
| Overlay grün/gelb/rot + Gesamtstatistik | **Noch nicht** |
| ROI-Datensatz 200–500/Split, 70/15/15, MobileNetV2 | **Noch nicht** (nur Kasten-YOLO-Datenpipeline) |

Die vier Ecken aus dem **YOLO-Rechteck** sind eine **Näherung** für Warp (nicht die exakte Physik-Oberfläche); für mehr Genauigkeit später echte Oberflächen-Ecken annotieren oder Pose/OBB erweitern.

## Projektstruktur

- `kasten.yaml` — Dataset-Konfiguration für `yolo train`
- `kastendetektion/detector.py` — **`detect_crate(frame)`** (YOLO primär, Canny/Kontur als Fallback)
- `kastendetektion/warp_grid.py` — **Warp Perspective** + **4×5 Slot-Mittelpunkte** (+ ROI-Helfer für spätere CNNs)
- `scripts/prepare_dataset.py` — flache `images/` + `labels/` → `train/` / `val/`
- `scripts/train_yolo.py` — Fine-Tuning (`epochs` Standard 50)
- `scripts/verify_labels.py` — prüft YOLO-Labels (Klasse 0, Werte in [0, 1])
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

## Daten von Dropbox

Trainingsbilder aus dem Kursordner:  
[Dropbox — Trainingsbilder](https://www.dropbox.com/scl/fo/0d5055swt00lm3oemtv30/AA5mTgKvz6-RsGlngpAuu_Y?rlkey=usqutwwa68w591k9ch2frowjw&e=1&st=m6kvlxcf&dl=0)

1. Ordner **`data/kasten_dataset/images/`** nutzen (liegt im Projekt; dort liegen auch `HINWEIS.txt` und `.gitkeep` — **eigene Fotos werden von Git ignoriert**).
2. Bilder labeln — siehe **[LabelImg Schritt für Schritt](#labelimg-schritt-für-schritt)** unten.
3. Labels prüfen:

```bash
python scripts/verify_labels.py
```

4. Split ausführen (im Repo-Root):

```bash
python scripts/prepare_dataset.py
```

5. Training:

```bash
python scripts/train_yolo.py --epochs 50
```

CLI analog zum Arbeitspaket:

```bash
yolo train model=yolov8n.pt data=kasten.yaml epochs=50
```

6. Inferenz mit Gewichten z. B.:

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
| Labels prüfen | `python scripts/verify_labels.py` |
| Label-Web-UI | `streamlit run scripts/label_ui.py` |

Voraussetzung für sinnvolles YOLO: zuerst Daten labeln und trainieren; bis dahin liefert die Pipeline den **Canny-Fallback** für erste Demos.
