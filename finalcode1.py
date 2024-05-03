import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk


class ExerciseTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gym Broach")

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.cap = cv2.VideoCapture(1)
        self.tracker = None
        self.tracker_running = False

        self.exercise_options = ["Bicep Curls", "Pushups", "Overhead Press", "Squats"]
        self.selected_exercise = tk.StringVar()
        self.selected_exercise.set(self.exercise_options[0])

        self.setup_gui()

    def setup_gui(self):
        exercise_label = ttk.Label(self.root, text="Select Exercise:")
        exercise_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")

        exercise_combobox = ttk.Combobox(self.root, textvariable=self.selected_exercise, values=self.exercise_options)
        exercise_combobox.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        start_button = ttk.Button(self.root, text="Start", command=self.start_tracking)
        start_button.grid(row=1, column=0, padx=10, pady=5)

        stop_button = ttk.Button(self.root, text="Stop", command=self.stop_tracking)
        stop_button.grid(row=1, column=1, padx=10, pady=5)

        self.video_label = ttk.Label(self.root)
        self.video_label.grid(row=2, column=0, columnspan=2)

    def start_tracking(self):
        if not self.tracker_running:
            exercise = self.selected_exercise.get()
            if exercise == "Bicep Curls":
                self.tracker = ExerciseTracker(self.cap, self.mp_drawing, self.mp_pose, "bicep_curls")
            elif exercise == "Pushups":
                self.tracker = ExerciseTracker(self.cap, self.mp_drawing, self.mp_pose, "pushups")
            elif exercise == "Overhead Press":
                self.tracker = ExerciseTracker(self.cap, self.mp_drawing, self.mp_pose, "overhead_press")
            elif exercise == "Squats":
                self.tracker = ExerciseTracker(self.cap, self.mp_drawing, self.mp_pose, "squats")

            self.tracker_running = True
            self.track()

    def track(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.tracker.track(frame)
            image = Image.fromarray(results)
            image = ImageTk.PhotoImage(image=image.resize((1280, 960)))
            self.video_label.configure(image=image)
            self.video_label.image = image

            if self.tracker_running:
                self.root.after(10, self.track)
        else:
            self.stop_tracking()

    def stop_tracking(self):
        self.tracker_running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.video_label.config(image=None)
        self.tracker = None


class ExerciseTracker:
    def __init__(self, cap, mp_drawing, mp_pose, exercise_type):
        self.cap = cap
        self.mp_drawing = mp_drawing
        self.mp_pose = mp_pose
        self.counter = 0
        self.stage = None
        self.warning = None
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.exercise_type = exercise_type

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def track(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = self.pose.process(frame)

        try:
            landmarks = results.pose_landmarks.landmark
            if self.exercise_type == "bicep_curls":
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = self.calculate_angle(shoulder, elbow, wrist)
            elif self.exercise_type == "pushups":
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                angle = self.calculate_angle(shoulder, elbow, wrist)

            elif self.exercise_type == "overhead_press":
                # Add overhead press tracking logic here
                pass
            elif self.exercise_type == "squats":
                left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                             landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                              landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # Calculate angles for both legs
                left_leg_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
                right_leg_angle = self.calculate_angle(right_hip, right_knee, right_ankle)

            if self.exercise_type == "squats":
                cv2.putText(image, str(f"{left_leg_angle:.2f}"),
                            tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, str(f"{right_leg_angle:.2f}"),
                            tuple(np.multiply(right_knee, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(image, str(f"{angle:.2f}"),
                            tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 2, cv2.LINE_AA)

            if self.exercise_type == "squats":
                if left_leg_angle < 75 and right_leg_angle < 75:
                    self.stage = "down"
                    self.warning = None
                if (left_leg_angle > 170 or right_leg_angle > 170) and self.stage == 'down':
                    self.stage = "up"
                    self.counter += 1
                    print(self.counter)
                if left_leg_angle in range(75,170) and right_leg_angle in range(75,170):
                    self.warning = "Half Rep Warning"
            elif self.exercise_type == "bicep_curls":
                if angle > 160:
                    self.stage = "down"
                    self.warning = None
                if angle < 35 and self.stage == 'down':
                    self.stage = "up"
                    self.counter += 1
                    print(self.counter)
                if angle < 90:
                    self.warning = "Half Rep Warning"
            else:
                if angle > 160:
                    self.stage = "down"
                    self.warning = None
                if angle < 30 and self.stage == 'down':
                    self.stage = "up"
                    self.counter += 1
                    print(self.counter)
                if angle < 90:
                    self.warning = "Half Rep Warning"

        except:
            pass

        cv2.rectangle(image, (0, 0), (300, 100), (245, 117, 16), -1)
        cv2.putText(image, 'REPS', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, 'STAGE', (110, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, self.stage,
                    (105, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        if self.warning is not None:
            cv2.putText(image, self.warning, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                       self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    root = tk.Tk()
    app = ExerciseTrackerGUI(root)
    root.mainloop()
