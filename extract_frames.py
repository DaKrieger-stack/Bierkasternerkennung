"""
extract_frames.py
-----------------
Liest Videos aus einem Ordner oder eine einzelne Videodatei,
und speichert Frames in regelmäßigen Abständen als JPG-Dateien.

Jedes Video bekommt einen eigenen Unterordner in --output, benannt nach dem
Videodateinamen (ohne Extension). Das erleichtert das spätere manuelle
Aussortieren: schlechtes Video → ganzen Unterordner löschen.

Verwendung:
    # Einzelne Datei:
    python scripts/extract_frames.py --input data/raw_videos/resistor_A_v1_01.mov --output data/frames/

    # Ganzer Ordner (verarbeitet alle Videos rekursiv):
    python scripts/extract_frames.py --input data/raw_videos/ --output data/frames/

    # Mit angepasster FPS:
    python scripts/extract_frames.py --input data/raw_videos/ --output data/frames/ --fps 3

Parameter:
    --input   Pfad zur Videodatei oder zum Ordner mit Videos
    --output  Zielordner für die extrahierten Frames
    --fps     Frames pro Sekunde die gespeichert werden (Standard: 2)
              2 fps bei 30-Sek-Video = ~60 Bilder pro Video.
              Nicht höher als 3-4 setzen: aufeinanderfolgende Frames
              werden zu ähnlich und blähen das Dataset sinnlos auf.
"""

import cv2
from pathlib import Path
import argparse
from tqdm import tqdm


# Rotation-Mapping: EXIF-Wert → OpenCV-Rotationskonstante
# iPhones speichern die Orientierung als Metadaten, OpenCV ignoriert das
# standardmäßig. Ohne diesen Fix kommen gedrehte Videos um 90° falsch raus.
ROTATION_MAP = {
    90:  cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def get_rotation_code(cap: cv2.VideoCapture):
    """Liest die EXIF-Orientierung aus dem Video und gibt den passenden
    OpenCV-Rotationscode zurück, oder None wenn keine Rotation nötig ist."""
    flag = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    return ROTATION_MAP.get(flag, None)


def extract_frames(video_path: Path, output_base_dir: Path, fps_target: float = 2.0) -> int:
    """Extrahiert Frames aus einem Video in gleichmäßigen Abständen.

    Frames landen in output_base_dir/<videoname>/ — ein Unterordner pro Video.
    Gibt die Anzahl gespeicherter Frames zurück.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"  FEHLER: Konnte {video_path.name} nicht öffnen.")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if video_fps <= 0:
        print(f"  WARNUNG: FPS konnte nicht gelesen werden für {video_path.name}, nehme 30 an.")
        video_fps = 30.0

    # Alle N-ten Frame speichern: bei 30fps Video und fps_target=2 → jeden 15. Frame
    frame_interval = max(1, int(round(video_fps / fps_target)))

    # Rotation korrigieren (iPhone-EXIF)
    rotation_code = get_rotation_code(cap)

    # Eigener Unterordner pro Video → leicht aussortierbar im Finder
    video_output_dir = output_base_dir / video_path.stem
    video_output_dir.mkdir(parents=True, exist_ok=True)

    base = video_path.stem  # z.B. "resistor_A_v1_01"
    idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Rotation anwenden falls nötig (kein Qualitätsverlust, nur Pixel-Umsortierung)
        if rotation_code is not None:
            frame = cv2.rotate(frame, rotation_code)

        if idx % frame_interval == 0:
            filename = f"{base}_frame_{saved:04d}.jpg"
            out_path = video_output_dir / filename
            # Qualität 95: guter Kompromiss — deutlich kleiner als unkomprimiert,
            # kein sichtbarer Qualitätsverlust. Für YOLO-Training vollkommen ausreichend.
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Extrahiert Frames aus Videos für YOLO-Training."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Videodatei oder Ordner mit Videos"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Zielordner für extrahierte Frames (Unterordner pro Video werden automatisch erstellt)"
    )
    parser.add_argument(
        "--fps", type=float, default=2.0,
        help="Frames pro Sekunde die gespeichert werden (Standard: 2.0)"
    )
    args = parser.parse_args()

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

    if args.input.is_dir():
        videos = sorted([
            f for f in args.input.rglob("*")
            if f.suffix.lower() in VIDEO_EXTENSIONS
        ])

        if not videos:
            print(f"Keine Videos in {args.input} gefunden.")
            return

        print(f"{len(videos)} Video(s) gefunden. Starte Extraktion mit {args.fps} fps...\n")
        total_saved = 0

        for video in tqdm(videos, desc="Videos verarbeiten", unit="Video"):
            saved = extract_frames(video, args.output, args.fps)
            tqdm.write(f"  {video.name}: {saved} Frames -> {args.output / video.stem}/")
            total_saved += saved

        print(f"\nFertig. Insgesamt {total_saved} Frames gespeichert in: {args.output}")

    elif args.input.is_file():
        if args.input.suffix.lower() not in VIDEO_EXTENSIONS:
            print(f"FEHLER: {args.input.name} ist kein unterstütztes Videoformat.")
            print(f"  Unterstützt: {', '.join(VIDEO_EXTENSIONS)}")
            return

        print(f"Extrahiere aus: {args.input.name}")
        saved = extract_frames(args.input, args.output, args.fps)
        print(f"\nFertig. {saved} Frames gespeichert in: {args.output / args.input.stem}/")

    else:
        print(f"FEHLER: {args.input} ist weder Datei noch Ordner.")


if __name__ == "__main__":
    main()
