import cv2
import mediapipe as mp
import numpy as np

from reps import RepCounter, calculate_angle

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

MIN_VISIBILITY = 0.5


def get_arm(landmarks, side):
    """Return (shoulder, elbow, wrist, visibility) for 'LEFT' or 'RIGHT'."""
    joints = [landmarks[mp_pose.PoseLandmark[f"{side}_{name}"].value]
              for name in ("SHOULDER", "ELBOW", "WRIST")]
    points = [[j.x, j.y] for j in joints]
    visibility = min(j.visibility for j in joints)
    return (*points, visibility)


def main():
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam. Check it isn't in use by another app "
              "and that camera access is allowed in Windows privacy settings.")
        return

    reps = RepCounter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB for MediaPipe, then back to BGR for OpenCV rendering
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        h, w, _ = image.shape
        angle = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Use whichever arm the camera sees best (side-on push-ups hide one arm)
            shoulder, elbow, wrist, vis = max(
                (get_arm(landmarks, "LEFT"), get_arm(landmarks, "RIGHT")),
                key=lambda arm: arm[3],
            )

            if vis > MIN_VISIBILITY:
                angle = calculate_angle(shoulder, elbow, wrist)

                reps.update(angle)

                elbow_px = tuple(np.multiply(elbow, [w, h]).astype(int))
                cv2.putText(image, f"{int(angle)}", elbow_px,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
            )

        # Stats panel
        cv2.rectangle(image, (0, 0), (260, 80), (40, 40, 40), -1)
        cv2.putText(image, "REPS", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, str(reps.count), (15, 68), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, "STAGE", (120, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(image, (reps.stage or "-").upper(), (120, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        if angle is None:
            cv2.putText(image, "Arm not visible", (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Calisthenics Tracker  (q: quit, r: reset)", image)

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            reps.reset()

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
