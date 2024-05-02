import cv2
import argparse
import numpy as np
from ultralytics import YOLO
import supervision as sv

DETECTION_AREA = np.array([
    [0, 0],
    [0.3, 0],
    [0.3, 0.5],
    [0, 0.5]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time YOLOv8 Object Detection")
    parser.add_argument(
        "--resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int,
        help="Resolution of the webcam input (width height)"
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    frame_width, frame_height = args.resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    yolo_detector = YOLO("yolov8l.pt")

    detection_area = (DETECTION_AREA * np.array(args.resolution)).astype(int)
    detection_zone = sv.PolygonZone(polygon=detection_area, frame_resolution_wh=tuple(args.resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=detection_zone, 
        color=sv.Color.green(),
        thickness=2,
        text_thickness=3,
        text_scale=1.5
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        result = yolo_detector(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{yolo_detector.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        annotated_frame = sv.BoxAnnotator(
            thickness=1,
            text_thickness=1,
            text_scale=1.2
        ).annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        detection_zone.trigger(detections=detections)
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)

        cv2.imshow("YOLOv8 Object Detection", annotated_frame)

        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
