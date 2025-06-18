# Face Recognition Celebrity

A simple Python-based celebrity face recognition program using the `face_recognition` library.  
It trains on celebrity images and recognizes faces in unknown images, showing the recognized celebrity name.

---

## Features

- Train on a folder of celebrity images organized by person’s name
- Save/load face encodings to speed up recognition
- Recognize faces in unknown images and label them with celebrity names
- Support for HOG (CPU) or CNN (GPU) face detection models

---

## Requirements

- Python 3.7+
- `face_recognition`
- `Pillow`
- `argparse`

You can install the dependencies using:

```bash
pip install -r requirements.txt
training/
├── drake/
│   ├── drake1.jpg
│   └── drake2.jpg
└── beyonce/
    ├── beyonce1.jpg
    └── beyonce2.jpg

python detector.py --train --model hog

python detector.py --test --file unknown.jpg --model hog
