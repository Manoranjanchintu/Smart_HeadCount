import cv2
import os
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load once globally

def detect_from_image(image_path):
    image = cv2.imread(image_path)
    results = model(image)[0]

    person_count = 0
    for box in results.boxes:
        cls = int(box.cls[0])
        if model.names[cls] == 'person':
            person_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, f'person {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    result_filename = os.path.basename(image_path)
    result_path = os.path.join('static/results', result_filename)
    cv2.imwrite(result_path, image)

    return result_filename, person_count


def detect_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    max_person_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        frame_count = 0
        for box in results.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == 'person':
                frame_count += 1

        max_person_count = max(max_person_count, frame_count)

    cap.release()
    return max_person_count


def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)[0]
        person_count = 0  # <-- Add this line to count persons

        for box in results.boxes:
            cls = int(box.cls[0])
            if model.names[cls] == 'person':
                person_count += 1  # <-- Count each detected person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, 'person', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 🟡 Draw the total person count on top-left of the frame
        cv2.putText(frame, f'Total Persons: {person_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
