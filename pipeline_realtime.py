import cv2
from ultralytics import YOLO
from PIL import Image
import numpy as np

# IOU threshold for non-max suppression
iou_threshold = 0.5

def Frame_Validation(input, valid_model):                 
    frame_valid = valid_model(input)
    x1, y1, x2, y2 = None, None, None, None 
    for result in frame_valid:
        classes = result.boxes.cls.tolist()
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0].tolist())

    if classes:           
        if classes[0] == 0.0:
            print("Khung bi nghieng")
        elif classes[0] == 1.0:
            print("Khung bi choi")
        elif classes[0] == 2.0:
            print("Khung bi che")
        elif classes[0] == 3.0:
            print("Khung bi sai")
    else:
        print("Khung OK")
    return x1, y1, x2, y2,classes

def NMS(boxes):
    selected_detections = []
    for current in boxes:
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
    return selected_detections

def corrected_number(number):
    if number < 10:
        return number
    elif 10 <= number < 20:
        return number - 10
    elif number >= 20:
        return number - 20

def compute_iou(box1, box2):
    # Compute the intersection over union (IoU) of two boxes
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou

def predict_image(image, frame_model, prediction_model, valid_model, iou_threshold=0.5):
    # Initialize a variable for the predicted label
    predicted_label = ""

    # Detect frames using the first model
    frame_results = frame_model(image)

    # Process each detected frame
    for frame_result in frame_results:
        detected_number = ""
        cropped_frame = None
        for bbox in frame_result.boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = bbox.xyxy[0].tolist()

            x1, y1, x2, y2 = map(int, bbox.xyxy[0].tolist())

            # Crop the image using the bounding box coordinates
            cropped_frame = frame[y1:y2, x1:x2]

            """ # Crop the image using the bounding box coordinates
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_image_array = np.array(cropped_image) """

            # Check if the frame is acceptable or not
            x3,y3,x4,y4, valid = Frame_Validation(cropped_frame, valid_model)

            prediction_results = prediction_model(cropped_frame)
            for result in prediction_results:
                boxes = result.boxes.xyxy.tolist()
                classes = result.boxes.cls.tolist()
                names = result.names
                confidences = result.boxes.conf.tolist()
                detections = list(zip(boxes, classes, confidences))
                boxes_sorted = sorted(detections, key=lambda x: x[0][0], reverse=False)

                #Non-Max Supression
                selected_detections = NMS(boxes_sorted)

                detected_number = ""
                for box, cls, conf in selected_detections:
                    name = names[int(cls)]
                    number = int(name)
                    corrected_num = corrected_number(number)
                    detected_number += str(corrected_num)

                    """ # Draw bounding box and label on the frame
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{name}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) """

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.rectangle(frame, ((x3), (y3)), ((x4), (y4)), (0, 255, 0), 2)
                    cv2.putText(frame, f"Predicted: {detected_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Predicted: {valid}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            for result in prediction_results:
                predicted_label += detected_number
            print(f"Processed: {predicted_label}")
    return predicted_label, cropped_frame

# Path to the model files
frame_model_path = "D:/HUST/dev/py/Digit_catkhung/model/khung/khung_best_1306.pt"
prediction_model_path = "D:/HUST/dev/py/Digit_catkhung/model/so/best_01_so.pt"
valid_model_path = "D:/HUST/dev/py/Digit_catkhung/model/khung hop le/best_2107.pt"

# Initialize YOLO models
frame_model = YOLO(frame_model_path)
prediction_model = YOLO(prediction_model_path)
valid_model = YOLO(valid_model_path)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Predict the labels on the frame and get the cropped frame
    predicted_labels, cropped_frame = predict_image(frame, frame_model, prediction_model, valid_model, iou_threshold)
    print(predicted_labels)

    """ # Display the cropped frame
    if cropped_frame is not None:
        cv2.imshow('Cropped Frame', cropped_frame) """

    # Display the original frame
    cv2.imshow('Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
