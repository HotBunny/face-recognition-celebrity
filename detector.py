import argparse
import pickle
from pathlib import Path
from collections import Counter

import face_recognition
from PIL import Image, ImageDraw, ImageFont

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
TRAINING_DIR = Path("training")
OUTPUT_DIR = Path("output")

BOUNDING_BOX_COLOR = (0, 255, 0)  # Green
TEXT_COLOR = (255, 255, 255)      # White
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # Adjust if needed


def train(model="hog", encodings_path=DEFAULT_ENCODINGS_PATH):
    print(f"[INFO] Training on images inside: {TRAINING_DIR}")
    names = []
    encodings = []

    for image_path in TRAINING_DIR.glob("*/*"):
        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        name = image_path.parent.name
        print(f"  Encoding {name} from {image_path.name}")
        image = face_recognition.load_image_file(image_path)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    OUTPUT_DIR.mkdir(exist_ok=True)
    with encodings_path.open("wb") as f:
        pickle.dump({"names": names, "encodings": encodings}, f)

    print(f"[INFO] Training complete: {len(encodings)} faces saved.")


def recognize(image_path, model="hog", encodings_path=DEFAULT_ENCODINGS_PATH):
    print(f"[INFO] Recognizing faces in {image_path}...")
    if not encodings_path.exists():
        print("[ERROR] No trained encodings found. Please run with --train first.")
        return

    with encodings_path.open("rb") as f:
        data = pickle.load(f)

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image, model=model)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)

    try:
        font = ImageFont.truetype(FONT_PATH, 20)
    except IOError:
        font = ImageFont.load_default()

    recognized_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(data["encodings"], face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = data["names"][first_match_index]

        recognized_names.append(name)

        # Draw bounding box
        draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR, width=3)

        # Use textbbox instead of textsize
        bbox = draw.textbbox((0, 0), name, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw label background
        draw.rectangle(
            ((left, bottom), (left + text_width + 10, bottom + text_height + 5)),
            fill=BOUNDING_BOX_COLOR
        )

        # Draw text
        draw.text((left + 5, bottom), name, fill=TEXT_COLOR, font=font)

    pil_image.show()

    print(f"[INFO] Recognized faces: {recognized_names}")


def main():
    parser = argparse.ArgumentParser(description="Celebrity Face Recognition")
    parser.add_argument("--train", action="store_true", help="Train on images in training folder")
    parser.add_argument("--test", action="store_true", help="Recognize faces in an image")
    parser.add_argument("--file", type=str, help="Path to the image file for recognition")
    parser.add_argument(
        "--model",
        choices=["hog", "cnn"],
        default="hog",
        help="Face detection model: 'hog' for CPU, 'cnn' for GPU (if available)",
    )
    args = parser.parse_args()

    if args.train:
        train(model=args.model)

    if args.test:
        if not args.file:
            print("[ERROR] Please provide --file <image_path> to test")
            return
        recognize(image_path=args.file, model=args.model)


if __name__ == "__main__":
    main()
