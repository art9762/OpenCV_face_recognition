# Simple OpenCV face recognition on Python

A local camera demo using OpenCV and dlib. Each detected face gets its own
descriptor from the original RGB frame; drawing labels does not alter model input.

## Setup and run

Use Python 3.10 or later and a desktop environment that can open OpenCV windows.
Creating the environment and installing dependencies does not start the camera:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py --help
```

`dlib` may need CMake, a C++ compiler, and platform-specific build tools when a
compatible wheel is unavailable. The script expects these model files next to
`main.py` by default:

- `shape_predictor_68_face_landmarks.dat`
- `dlib_face_recognition_resnet_model_v1.dat`

Start camera capture explicitly:

```sh
python main.py
# Optional camera, model location, and database location:
python main.py --camera 1 --models-dir /path/to/models --database /path/to/faces.pkl
```

The default database is `known_faces.pkl` next to `main.py`, regardless of the
current working directory. Its parent directory must already exist and be writable.
Missing models or an invalid database are reported before camera capture begins.
`--tolerance` controls the match distance (default `0.6`); the nearest descriptor
strictly below this threshold is displayed. This threshold is not a calibrated
confidence score or a guarantee of identity.

## Register a face

Press **s** while exactly one face is visible, then enter its name in the terminal.
The app saves that displayed frame's descriptor. Zero or multiple faces block
registration, so a previous frame or another person's descriptor cannot be saved
accidentally. Reusing a name requires confirmation before replacement. Press
**q** in the OpenCV window or **Ctrl+C** in the terminal to exit.

The existing pickle dictionary format remains supported, including legacy NumPy
vectors; new descriptors are stored as tuples of numbers. Writes use atomic file
replacement. Only load databases you created and trust: pickle can execute code
while loading. To start a separate database, supply a new `--database` path.
Keep generated databases private. New `*.pkl` files are ignored by Git, but the
pre-existing tracked database remains tracked and should not be committed after
registering personal data.

## Offline checks

These tests need only Python's standard library. They never open a camera, load
dlib models, read the repository's face database, or inspect images:

```sh
python3 -m unittest discover -s tests -v
```

They exercise synthetic per-face descriptors, RGB conversion/annotation order,
single-face registration, database validation, failed saves, and camera cleanup
using fakes. Real-camera compatibility, recognition quality, and model performance
still require a separate manual test with consented images or participants.
