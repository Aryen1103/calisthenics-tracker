# Calisthenics Tracker

Real-time rep counter for calisthenics and arm exercises using a webcam. It tracks body pose with [MediaPipe Pose](https://github.com/google-ai-edge/mediapipe), calculates the elbow angle every frame, and counts a rep each time the arm bends and fully extends again. It runs on a laptop CPU with no GPU needed.

## How it works

Each video frame goes through this pipeline:

1. Capture from the webcam with OpenCV
2. Convert BGR → RGB for MediaPipe
3. Detect 33 body landmarks with MediaPipe Pose
4. Take the shoulder, elbow and wrist of whichever arm is most visible
5. Calculate the elbow angle with NumPy
6. Update the rep counter and draw the skeleton, angle and stats onto the frame

A rep is counted when the elbow angle drops below **90°** and then returns above **160°**.

## Supported exercises

The tracker measures the elbow, so it works for push-ups, pull-ups, dips, bicep curls and overhead tricep extensions.

Stand or sit **side-on** to the camera. Viewed from the front, the arm's bend is foreshortened and reps get missed.

## Setup

Requires Python 3.9–3.12.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
python tracker.py
```

To use another camera, or count reps in a recorded video instead of a live
webcam, pass `--source`:

```bash
python tracker.py --source 1               # second camera
python tracker.py --source pushups.mp4     # video file
```

The final rep count is printed when the window closes or the video ends.

> `mediapipe` is pinned to `0.10.14`. MediaPipe 1.x removed the `mp.solutions` API this project uses.

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Reset rep counter |

## Configuration

To tune rep detection, edit the thresholds in `reps.py`:

```python
UP_ANGLE = 160        # arm counts as extended above this
DOWN_ANGLE = 90       # arm counts as bent below this
```

and the landmark confidence cut-off in `tracker.py`:

```python
MIN_VISIBILITY = 0.5  # ignore the arm when landmark confidence is lower
```

The gap between the two angles is deliberate: landmark positions jitter by a
few degrees from frame to frame, and a single threshold would count that
jitter as extra reps.

## Tests

The angle calculation and rep counting live in `reps.py`, which depends only on
NumPy, so they can be tested without a webcam:

```bash
pip install -r requirements-dev.txt
python -m pytest
```
