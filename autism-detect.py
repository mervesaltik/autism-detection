import cv2
import mediapipe as mp
import numpy as np
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import time
from collections import deque
import tkinter as tk
from tkinter import filedialog
from threading import Thread


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

MOVEMENT_THRESHOLD = 0.18
prev_landmarks = None
static_frame_count = 0
total_frame_count = 0

movement_threshold = 5
movement_counter = 0
last_nose_position = None
last_autism_probability = 0.0
start_time = time.time()

def detect_blinks(frame, detector, eye_points, ratio_list, counter, blink_count, start_time, blink_time,
                  blink_timestamp, face_detect_time):
    img, faces = detector.findFaceMesh(frame, draw=False)


    if faces:
        face_detect_time = time.time()

    if faces:
        face = faces[0]
        for id in eye_points:
            cv2.circle(img, face[id], 3, (0, 255, 0), -1)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]

        lenght_Hor, _ = detector.findDistance(leftUp, leftDown)
        lenght_Ver, _ = detector.findDistance(leftLeft, leftRight)

        cv2.line(img, leftLeft, leftRight, (255, 0, 255), 3)

        cv2.line(img, leftUp, leftDown, (255, 0, 255), 3)
        ratio = int((lenght_Hor / lenght_Ver) * 100)

        ratio_list.append(ratio)
        if len(ratio_list) > 10:
            ratio_list.pop(0)
        ratio_avg = sum(ratio_list) / len(ratio_list)

        if ratio_avg < 35 and counter == 0:
            blink_count += 1
            counter = 1
            start_time = time.time()
            blink_timestamp.append(time.time())

        if counter != 0:
            counter += 1
            if counter > 10:
                end_time = time.time()
                duration = end_time - start_time
                blink_time.append(round(duration, 2))
                counter = 0
        cvzone.putTextRect(img, f"Blink:{blink_count}", (100, 200))

    return img, ratio_list, counter, blink_count, start_time, blink_time, blink_timestamp, face_detect_time

def analyze_blink_duration_and_ratio(blink_time, ratio_list, blink_timestamp, video_duration, face_detect_time):
    avg_blink_duration = sum(blink_time) / len(blink_time) if blink_time else 0
    avg_ratio = sum(ratio_list) / len(ratio_list) if ratio_list else 0

    blink_per_minute = len(blink_timestamp) * (60 / video_duration) if video_duration > 0 else 0

    face_detect_ratio = face_detect_time / video_duration if video_duration > 0 else 0

    print(f"Average blink duration: {avg_blink_duration:.2f} seconds")
    print(f"Average blink ratio: {avg_ratio:.2f}")
    print(f"Blink count per minute: {blink_per_minute:.2f}")
    print(f"Face detection ratio: {face_detect_ratio:.2f}")

    autism_probability = 0

    if blink_per_minute > 15 and avg_blink_duration > 0.4:
        autism_probability = 55 + ((avg_blink_duration - 0.4) / (1.0 - 0.4)) * 30
    elif 10 <= blink_per_minute <= 15 and avg_blink_duration > 0.4:
        autism_probability = 30 + ((avg_blink_duration - 0.4) / (1.0 - 0.4)) * 20
    elif blink_per_minute < 10 and avg_blink_duration > 0.4:
        autism_probability = 40 + ((avg_blink_duration - 0.4) / (1.0 - 0.4)) * 20
    elif blink_per_minute > 15 and 0.1 <= avg_blink_duration <= 0.4:
        autism_probability = 40 + ((avg_blink_duration - 0.1) / (0.4 - 0.1)) * 20
    elif 10 <= blink_per_minute <= 15 and avg_blink_duration < 0.1:
        autism_probability = 20 + ((avg_blink_duration) / 0.1) * 20
    elif blink_per_minute > 15 and avg_blink_duration < 0.1:
        autism_probability = 40 + ((avg_blink_duration) / 0.1) * 20
    elif blink_per_minute < 10 and avg_blink_duration < 0.1:
        autism_probability = 10 + ((avg_blink_duration) / 0.1) * 30
    elif blink_per_minute < 10 and 0.1 <= avg_blink_duration <= 0.4:
        autism_probability = 20 + ((avg_blink_duration - 0.1) / (0.4 - 0.1)) * 20
    elif 10 <= blink_per_minute <= 15 and 0.1 <= avg_blink_duration <= 0.4:
        autism_probability = 10 + ((avg_blink_duration - 0.1) / (0.4 - 0.1)) * 20

    autism_probability = min(max(autism_probability, 0), 100)

    if autism_probability > 0:
        print(f"Potential autism signs detected. Autism probability: {autism_probability:.2f}%")
        autism_text = f"Eyes: {autism_probability:.2f}%"
    else:
        print("Normal blinking behavior detected. Autism probability: 0%")
        autism_text = "Eyes: 0%"

    return autism_text

def calculate_movement(curr_landmarks, previous_landmarks):
    if previous_landmarks is None:
        return 0
    return np.linalg.norm(curr_landmarks - previous_landmarks, axis=1).mean()

def calculate_autism_ratio(static_frame_cntr, total_frame_cntr):
    if total_frame_cntr > 0:
        return (static_frame_cntr / total_frame_cntr) * 100
    return 0

def calculate_autism_probability(movement_counter, time_elapsed, last_autism_probability):
    movement_rate = movement_counter / time_elapsed
    new_autism_probability = movement_rate * 15
    autism_probability = last_autism_probability * 0.9 + new_autism_probability * 0.1
    return min(autism_probability, 70.0)

def process_video(video_path, probability_label_hands, probability_label_face):
    global static_frame_count, total_frame_count, prev_landmarks, movement_counter, last_nose_position, last_autism_probability

    cap = cv2.VideoCapture(video_path)

    detector = FaceMeshDetector(maxFaces=1)
    video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    LEFT_EYE = [22, 23, 24, 26, 110, 157, 158, 159, 160, 130, 243]
    ratio_list = deque([], maxlen=10)
    blink_Count = 0
    counter = 0
    start_time = 0
    blink_time = []
    blink_timestamp = []
    face_detect_time = 0
    if not cap.isOpened():
        print("Video cannot open.")
        return

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            frame, ratio_list, counter, blink_Count, start_time, blink_time, blink_timestamp, face_detect_time = detect_blinks(
                frame, detector, LEFT_EYE, ratio_list, counter, blink_Count, start_time, blink_time, blink_timestamp,
                face_detect_time)

            autism_text = analyze_blink_duration_and_ratio(blink_time, ratio_list, blink_timestamp, video_duration,
                                                           face_detect_time)

            blink_label.config(text=f"Blink Count: {blink_Count}")
            autism_label.config(text=autism_text)

            cvzone.putTextRect(frame, autism_text, (100, 100), scale=1.2)

            total_frame_count += 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results_hands = hands.process(frame_rgb)
            if results_hands.multi_hand_landmarks:
                for hand_landmarks in results_hands.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    current_landmarks = np.array(
                        [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

                    movement = calculate_movement(current_landmarks, prev_landmarks)

                    if movement < MOVEMENT_THRESHOLD:
                        static_frame_count += 1

                    prev_landmarks = current_landmarks

            autism_ratio_hands = calculate_autism_ratio(static_frame_count, total_frame_count)

            results_face = face_mesh.process(frame_rgb)
            if results_face.multi_face_landmarks:
                face_landmarks = results_face.multi_face_landmarks[0]
                nose_x = face_landmarks.landmark[1].x * frame.shape[1]
                nose_y = face_landmarks.landmark[1].y * frame.shape[0]
                nose_position = (nose_x, nose_y)

                if last_nose_position is not None:
                    nose_change = np.linalg.norm(np.array(nose_position) - np.array(last_nose_position))
                    if nose_change > movement_threshold:
                        movement_counter += 1

                last_nose_position = nose_position

                time_elapsed = time.time() - start_time
                autism_ratio_face = calculate_autism_probability(movement_counter, time_elapsed,
                                                                 last_autism_probability)
                last_autism_probability = autism_ratio_face

                probability_label_face.config(text=f"Face: {autism_ratio_face:.2f}%")

            probability_label_hands.config(text=f"Hands: {autism_ratio_hands:.2f}%")

            cv2.imshow("Video Isleme", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def select_video(probability_label_hands, probability_label_face,blink_label,autism_label):
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", ".mp4;.avi;*.mov")])
    if video_path:
        thread = Thread(target=process_video, args=(video_path, probability_label_hands, probability_label_face))
        thread.start()

root = tk.Tk()
root.title("Autism Analysis")
root.geometry("400x300")

label = tk.Label(root, text="Autism Analysis", font=("Arial", 16))
label.pack()

probability_label_hands = tk.Label(root, text="Hand: 0%", font=("Arial", 14))
probability_label_hands.pack()

probability_label_face = tk.Label(root, text="Face: 0%", font=("Arial", 14))
probability_label_face.pack()

blink_label = tk.Label(root, text="Blink Count: 0", font=("Arial", 14))
blink_label.pack()

autism_label = tk.Label(root, text="Eyes: 0%", font=("Arial", 14))
autism_label.pack()

select_button = tk.Button(root, text="Select Video", font=("Arial", 12), command=lambda: select_video(probability_label_hands,probability_label_face,blink_label, autism_label))
select_button.pack()

exit_button = tk.Button(root, text="Exit", command=root.quit, font=("Arial", 12))
exit_button.pack()

root.mainloop()


