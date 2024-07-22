from ultralytics import YOLO
import cv2

model = YOLO('D:/HUST/dev/py/Digit_catkhung/model/khung/khung_best_1306.pt')
cap = cv2.VideoCapture(0)
iou_threshold = 0.5

def corrected_number(number):
    if number < 10:
        return number
    elif 10 <= number < 20:
        return number - 10
    elif number >= 20:
        return number - 20

def compute_iou(box1, box2):
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    results = model(frame)

    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    detections = list(zip(boxes, classes, confidences))

    # Sort boxes by their x-coordinate (left to right)
    boxes_sorted = sorted(detections, key=lambda x: x[0][0], reverse=False)

    # Non-Maximum Suppression (NMS) to remove overlapping boxes
    selected_detections = []
    for current in boxes_sorted:
        current_box, current_class, current_conf = current
        keep = True
        for selected in selected_detections:
            selected_box, selected_class, selected_conf = selected
            if compute_iou(current_box, selected_box) > iou_threshold:
                if current_conf > selected_conf:
                    selected_detections.remove(selected)
                else:
                    keep = False
                    break
        if keep:
            selected_detections.append(current)

    detected_number = ""
    for box, cls, conf in selected_detections:
        x1, y1, x2, y2 = box
        name = names[int(cls)]
        number = name

        confidence = conf
        detected_number += str(name)

        if conf > 0.25:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f"{name}: {confidence:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Predicted: {detected_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('YOLO V8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
