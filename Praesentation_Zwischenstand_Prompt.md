# Prompt für Claude – Zwischenstand-Präsentation (02. Juni)

> **So nutzt du diese Datei:** Den kompletten Block unter „PROMPT" markieren und an Claude
> schicken. Lade dabei am besten die Bilder aus `demo_output/` als Anhang mit hoch (oder
> ersetze sie vorher durch echte Fotos). Claude soll daraus eine PowerPoint mit **max. 4 Folien**
> erzeugen.

---

## Kontext (für dich, nicht Teil des Prompts)

- **Anlass:** Erster Labortermin nach der Pfingstpause (02. Juni). Verlangt wird ein **Zwischenstand**,
  präsentiert als **PowerPoint-Folien** zusätzlich zur Software.
- **Harte Vorgaben aus der Mail:** **max. 4 Folien pro Gruppe**, **ca. 5 Minuten Redezeit**.
  Inhalt: anhand des Konzeptplans **kurz erklären, was/wie/warum umgesetzt** wurde, **wo es noch
  Schwierigkeiten gibt**, und **was im Juni noch umgesetzt** werden soll. Abschlusspräsentation am 23. Juni.
- **Gruppe:** Mia Scharpf (223488), David Krieger (223493), Robin Müller (222919) – AKI.
- **Demo-Bilder:** Liegen in `demo_output/` (aktuell **synthetischer** Testkasten, da noch keine
  echten Fotos gelabelt sind – siehe Hinweis im Prompt).

---

## PROMPT

```
Du bist ein Experte für technische Präsentationen. Erstelle eine deutschsprachige
Zwischenstand-Präsentation für ein Hochschul-Computer-Vision-Projekt.

# Rahmenbedingungen (zwingend einhalten)
- MAXIMAL 4 inhaltliche Folien (plus optional eine schlichte Titelzeile auf Folie 1).
- Vortragsdauer ca. 5 Minuten -> pro Folie wenige, prägnante Stichpunkte, KEINE Textwüsten.
- Sprache: Deutsch. Sachlich, technisch korrekt, studentisch-professionell.
- Es ist ein ZWISCHENSTAND (ca. 50% des Labors absolviert), kein fertiges Produkt.
- Liefere am Ende: (a) ein lauffähiges python-pptx-Skript, das die .pptx erzeugt, UND
  (b) pro Folie kompakte Speaker-Notes für insgesamt ~5 Minuten Redezeit.
- Erfinde KEINE Ergebnisse/Metriken. Nutze nur die unten genannten Fakten.

# Projekt
Titel: "Bierkasten-Erkennung – Hybridansatz (Klassische Bildverarbeitung + Machine Learning)".
Team: Mia Scharpf, David Krieger, Robin Müller (Studiengang AKI).
Ziel: Python-Anwendung zur Echtzeit-Erkennung des Füllzustands eines Bierkastens über Kamera.
Zwei Fragen: (1) Ist der Kasten vollständig befüllt (Flasche vorhanden / Slot leer)?
(2) Sind die Flaschen voll oder leer (Kronkorken vorhanden = voll)?
Prototyp: Paulaner Hefe Weißbier, 20 Flaschen im 4x5-Raster.

# Methodik (Konzeptplan, "warum")
Hybrid aus klassischer CV (OpenCV) für Geometrie + ML für Semantik.
Begründung: Klassische Methoden sind schnell, deterministisch, brauchen keine Trainingsdaten;
ML ist robuster gegen Licht, Reflexionen und Perspektive.
Pipeline (sequenziell pro Frame):
1. Bildaufnahme (OpenCV Kamera-Stream)
2. Kastendetektion: YOLOv8 (primär) + Canny/findContours (Fallback)
3. Perspektivische Entzerrung (Warp Perspective)
4. Grid Mapping: 4x5 -> 20 Slot-Mittelpunkte
5. Stufe 1: Slot-Klassifikation (vorhanden/leer)
6. Stufe 2: Deckel-Klassifikation (Kronkorken voll/leer)
7. Visualisierung: farbcodiertes Overlay (grün=voll, gelb=leer, rot=fehlt) + Gesamtstatistik

# AKTUELLER STAND – bereits umgesetzt und getestet ("was" + "wie")
- Kamera-Stream live mit Overlay (cv2.VideoCapture).
- Kastendetektion zweigleisig:
  * Primär: Ultralytics YOLOv8 (Fine-Tuning auf eigene Klasse "bierkasten" vorbereitet:
    Dataset-Config, Label-Tooling, Trainings-Skript).
  * Fallback: klassisch (Canny + findContours + minAreaRect) -> liefert sofort Ergebnisse OHNE Training.
  * Ausgabe der Schnittstelle: detect_crate(frame) -> Bounding Box (x,y,w,h) + 4 Eckpunkte
    (Reihenfolge oben-links, oben-rechts, unten-rechts, unten-links) + Konfidenz + Orientierung.
- Zusätzlicher, stärkerer klassischer Detektor: HSV-Rotsegmentierung + Otsu + Morphologie
  + Hough-Linien/Kreise + Hu-Momente-Scoring (robuste Auswahl des Kasten-Rechtecks).
- Perspektivische Entzerrung (Warp Perspective) auf normierte Draufsicht inkl. 4x5-Grid-Mapping
  (20 Slot-Mittelpunkte), Rückprojektion ins Originalbild und ROI-Extraktion je Slot
  (Vorbereitung für die Stufe-1-Klassifikation). -> Das ist bereits Vorarbeit für den nächsten Termin.
- Saubere, dokumentierte Schnittstelle zwischen Detektion und Grid-Modul.
- Komplette Daten-/Trainings-Pipeline steht: lokale Label-UI (Streamlit) + LabelImg-Workflow,
  Label-Prüfung, Train/Val-Split, YOLOv8-Trainingsskript (CPU/GPU-Defaults, AutoBatch, Early Stopping).
- End-to-End-Smoke-Test bestätigt: Detektion -> Entzerrung -> 20 Slot-Mittelpunkte funktioniert
  (auf synthetischem Testbild, da noch keine echten Fotos vorliegen).

# SCHWIERIGKEITEN / OFFENE PUNKTE ("wo hakt es")
- Noch keine eigenen Trainingsbilder aufgenommen/gelabelt -> YOLO läuft aktuell nur über den
  klassischen Fallback; trainierte Gewichte (best.pt) und das Ziel mAP > 0.8 stehen noch aus.
- YOLO liefert nur eine achsparallele Box -> die 4 Eckpunkte für die Entzerrung sind eine
  Näherung (keine echten Oberflächen-Ecken; Pose/OBB noch nicht umgesetzt). Orientierung
  liefert bislang nur der klassische Pfad.
- Hauptrisiken laut Risikoanalyse: wechselnde Lichtverhältnisse und Reflexionen auf Glas/Kronkorken.
- Stufe 1 (Slot voll/leer) und Stufe 2 (Kronkorken) sind konzipiert, aber noch nicht implementiert.

# AUSBLICK JUNI – bis zur Abschlusspräsentation (23. Juni)
- Trainingsbilder aufnehmen (verschiedene Winkel, Licht, Rotation) + labeln -> YOLOv8 fine-tunen,
  best.pt erzeugen, mAP > 0.8 erreichen.
- Stufe 1: Slot-Klassifikation (klassisch: Hough-Kreise/Helligkeit; ML: CNN/MobileNetV2 oder HOG+SVM).
- Stufe 2: Kronkorken-Erkennung (HSV-Fallback + CNN, robust gegen Reflexionen).
- Farbcodiertes Overlay (grün/gelb/rot) + Gesamtstatistik ("15 von 20 Flaschen, davon 12 voll").
- Frame-Stabilisierung (z. B. 10 stabile Frames) gegen Flackern.
- Genauere Entzerrung über echte Oberflächen-Ecken / Oriented Bounding Box.

# Empfohlene Folienstruktur (genau 4 inhaltliche Folien)
Folie 1 – Projekt & Konzept: Ziel, zwei Fragestellungen, Hybridansatz + warum, 7-Schritt-Pipeline (als Grafik/Liste).
Folie 2 – Umgesetzt (Termin-Fokus): Kastendetektion (YOLO + klassischer Fallback) und perspektivische
          Entzerrung + 4x5-Grid. Hier 1-2 Demo-Bilder einbinden (siehe unten).
Folie 3 – Schwierigkeiten & Erkenntnisse: offene Punkte, Risiken (Licht/Reflexion), Näherung bei den Eckpunkten.
Folie 4 – Ausblick Juni: Meilensteine bis 23. Juni (Training/mAP, Stufe 1, Stufe 2, Overlay+Statistik).

# Bildmaterial (falls als Anhang mitgegeben)
Aus dem Ordner demo_output/ (aktuell SYNTHETISCHER Testkasten – im Vortrag bitte als
"schematischer/synthetischer Test" kennzeichnen, da echte Fotos noch folgen):
- synthetic_crate.png   – Eingangsbild (synthetischer roter Kasten, leicht gedreht)
- 01_detection.png      – erkannte Bounding Box + 4 Eckpunkte (klassischer Fallback)
- 02_warped.png         – perspektivisch entzerrte Draufsicht
- 03_grid.png           – 4x5-Gitter mit 20 Slot-Mittelpunkten auf der Draufsicht
- 04_grid_on_original.png – 20 Slot-Mittelpunkte zurückprojiziert ins Originalbild
Empfehlung: Folie 2 mit 01_detection.png und 04_grid_on_original.png bestücken (zeigt Detektion -> Grid).
Wenn keine Bilder mitgegeben werden, beschreibe in den Folien klar, welche Visualisierung an welche Stelle gehört.

# Designvorgaben
- Klares, ruhiges Layout; gut lesbare Schrift; konsistente Farben (z. B. grün/gelb/rot als Legende auf Folie 1 oder 4).
- Pro Folie eine prägnante Überschrift + max. 4-5 Bulletpoints.
- Keine vollständigen Sätze in den Folien; ausführliche Erklärungen gehören in die Speaker-Notes.

Erzeuge jetzt das python-pptx-Skript und die Speaker-Notes.
```
