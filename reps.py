"""Rep detection, kept free of OpenCV and MediaPipe so it can be tested on its own."""
import numpy as np

# Elbow angle thresholds for a rep (push-ups, pull-ups, dips)
UP_ANGLE = 160    # arm extended
DOWN_ANGLE = 90   # arm bent


def calculate_angle(a, b, c):
    """Angle at point b (in degrees) formed by points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return 360 - angle if angle > 180 else angle


class RepCounter:
    """Counts a rep each time the arm bends below DOWN_ANGLE and then extends above UP_ANGLE.

    The gap between the two thresholds stops landmark jitter around one value
    from counting extra reps.
    """

    def __init__(self, up_angle=UP_ANGLE, down_angle=DOWN_ANGLE):
        self.up_angle = up_angle
        self.down_angle = down_angle
        self.reset()

    def reset(self):
        self.count = 0
        self.stage = None

    def update(self, angle):
        """Feed one frame's elbow angle. Returns True when that frame completes a rep."""
        if angle < self.down_angle:
            self.stage = "down"
        elif angle > self.up_angle:
            completed = self.stage == "down"
            self.stage = "up"
            if completed:
                self.count += 1
                return True
        return False
