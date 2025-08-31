from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import time
from utils.speed_utils import calculate_speed


def process_video(input_path, output_path=None, yolo_model_path='models/yolov8m.pt'):
    # Load YOLOv8 model
    model = YOLO(yolo_model_path)
    # Initialize DeepSORT tracker
    tracker = DeepSort(max_age=30)
    # Load video
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    if output_path:
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_width, frame_height)
        )
    # Data structure for track history
    player_positions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, conf=0.5, verbose=False)[0]
        detections = []
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if int(class_id) == 0:  # Person class
                detections.append(([x1, y1, x2 - x1, y2 - y1], score, "person"))

        # Run DeepSORT tracker
        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            cx = int((l + r) / 2)
            cy = int((t + b) / 2)
            # Calculate speed
            speed = calculate_speed(track_id, cx, cy, player_positions)
            # Draw bounding box and speed
            cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 2)
            cv2.putText(
                frame, f"ID:{track_id} Speed:{int(speed)} px/s",
                (int(l), int(t) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )
        # Write frame to output and display
        if out: out.write(frame)
        cv2.imshow("Football Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # cleanup
    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video('videos/football.mp4', output_path='output/output.mp4')
