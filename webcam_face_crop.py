"""
Deliverable: webcam -> face detection -> face crop -> emotion model hook.

Run:
    python webcam_face_crop.py

Useful options:
    python webcam_face_crop.py --save-crops
    python webcam_face_crop.py --camera 1
    python webcam_face_crop.py --emotion-model path/to/model.h5

Controls:
    q = quit
    s = save the current face crop
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None  # type: ignore[assignment]

try:
    import numpy as np
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]


FaceBox = Tuple[int, int, int, int]


def require_runtime() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if np is None:
        missing.append("numpy")
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(f"Missing package(s): {packages}. Install with: pip install -r requirements_member1_webcam.txt")


class EmotionModel:
    """
    Small adapter for the next team member.

    If a Keras/TensorFlow model is passed with --emotion-model, this class sends
    a normalized grayscale face crop to that model. Otherwise it returns a clear
    placeholder result so the webcam and crop pipeline remains fully runnable.
    """

    labels = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")

    def __init__(self, model_path: Optional[str], input_size: int) -> None:
        self.model = None
        self.input_size = input_size

        if model_path:
            try:
                from tensorflow.keras.models import load_model

                self.model = load_model(model_path)
                print(f"[INFO] Loaded emotion model: {model_path}")
            except Exception as exc:
                print(f"[WARN] Could not load emotion model: {exc}")
                print("[WARN] Continuing with placeholder emotion output.")

    def predict(self, face_bgr: np.ndarray) -> str:
        if self.model is None:
            return "emotion model not connected"

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.input_size, self.input_size))
        normalized = resized.astype("float32") / 255.0
        batch = np.expand_dims(normalized, axis=(0, -1))

        probabilities = self.model.predict(batch, verbose=0)[0]
        label_index = int(np.argmax(probabilities))
        confidence = float(probabilities[label_index])
        label = self.labels[label_index] if label_index < len(self.labels) else str(label_index)
        return f"{label} ({confidence:.2f})"


def ensure_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)


def largest_face(faces: np.ndarray) -> Optional[FaceBox]:
    if faces is None or len(faces) == 0:
        return None
    return tuple(int(value) for value in max(faces, key=lambda face: face[2] * face[3]))


def crop_face(frame: np.ndarray, box: FaceBox, padding: float = 0.15) -> np.ndarray:
    """Crop the detected face box with a little padding, clipped to the frame."""
    x, y, w, h = box
    frame_h, frame_w = frame.shape[:2]

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame_w, x + w + pad_x)
    y2 = min(frame_h, y + h + pad_y)

    return frame[y1:y2, x1:x2].copy()


def draw_face_preview(frame: np.ndarray, face_crop: np.ndarray) -> None:
    """Show the latest crop in the corner of the webcam frame."""
    preview_size = 140
    preview = cv2.resize(face_crop, (preview_size, preview_size))

    y1, x1 = 10, frame.shape[1] - preview_size - 10
    y2, x2 = y1 + preview_size, x1 + preview_size

    frame[y1:y2, x1:x2] = preview
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.putText(
        frame,
        "Face Crop",
        (x1, y2 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def save_crop(output_dir: Path, face_crop: np.ndarray, index: int) -> Path:
    ensure_dir(output_dir)
    crop_path = output_dir / f"face_crop_{index:06d}.png"
    cv2.imwrite(str(crop_path), face_crop)
    return crop_path


def run_webcam(args: argparse.Namespace) -> None:
    require_runtime()

    cascade_path = args.face_cascade or os.path.join(
        cv2.data.haarcascades,
        "haarcascade_frontalface_default.xml",
    )
    face_detector = cv2.CascadeClassifier(cascade_path)
    if face_detector.empty():
        raise FileNotFoundError(f"Could not load OpenCV face cascade: {cascade_path}")

    emotion_model = EmotionModel(args.emotion_model, args.model_input_size)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open webcam with camera index {args.camera}")

    output_dir = Path(args.output_dir)
    crop_index = 0
    last_save_time = 0.0

    print("[INFO] Webcam started. Press q to quit, s to save current crop.")

    while True:
        ok, frame = camera.read()
        if not ok:
            print("[WARN] Could not read webcam frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        face_box = largest_face(faces)
        face_crop = None
        emotion_text = "No face detected"

        if face_box is not None:
            x, y, w, h = face_box
            face_crop = crop_face(frame, face_box, args.padding)
            emotion_text = emotion_model.predict(face_crop)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)
            draw_face_preview(frame, face_crop)

            if args.save_crops and time.time() - last_save_time >= args.save_every:
                crop_index += 1
                crop_path = save_crop(output_dir, face_crop, crop_index)
                last_save_time = time.time()
                print(f"[INFO] Saved face crop: {crop_path}")

        cv2.putText(
            frame,
            f"Emotion: {emotion_text}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Webcam -> Face Detection -> Face Crop", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s") and face_crop is not None:
            crop_index += 1
            crop_path = save_crop(output_dir, face_crop, crop_index)
            print(f"[INFO] Saved face crop: {crop_path}")

    camera.release()
    cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Webcam face detection and face crop pipeline.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index, usually 0.")
    parser.add_argument("--output-dir", default="face_crops", help="Folder for saved face crops.")
    parser.add_argument("--save-crops", action="store_true", help="Automatically save detected face crops.")
    parser.add_argument("--save-every", type=float, default=1.0, help="Seconds between automatic saved crops.")
    parser.add_argument("--padding", type=float, default=0.15, help="Extra padding around the face crop.")
    parser.add_argument("--face-cascade", default=None, help="Optional custom OpenCV Haar cascade XML path.")
    parser.add_argument("--emotion-model", default=None, help="Optional Keras/TensorFlow emotion model path.")
    parser.add_argument("--model-input-size", type=int, default=48, help="Emotion model image input size.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run_webcam(args)
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
