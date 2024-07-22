from ultralytics import YOLO
from PIL import Image
import numpy as np

iou_threshold = 0.5

def Frame_Validation(input, valid_model):                 
    frame_valid = valid_model(input)
    names = []

    for result in frame_valid:
        boxes = result.boxes.xyxy.tolist()
        classes = result.boxes.cls.tolist()
        names = result.names
        confidences = result.boxes.conf.tolist()
        detections = list(zip(boxes, classes, confidences))
        boxes_sorted = sorted(detections, key=lambda x: x[0][0], reverse=False)
        
        selected_detections = NMS(boxes_sorted)
        for box, cls, conf in selected_detections:
            name = names[int(cls)]
            number = int(name)
        #show predicted frame        
        result.show()           
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
    return classes

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

def predict_image(image_path, frame_model_path, prediction_model_path, Valid_model):
    # Initialize YOLO models
    frame_model = YOLO(frame_model_path)
    prediction_model = YOLO(prediction_model_path)
    Valid_model = YOLO(Valid_model)
    
    predicted_label = ""

    # Detect frames
    frame_results = frame_model(image_path)
    image = Image.open(image_path)

    # Process each detected frame
    for frame_result in frame_results:
        detected_number = ""
        for bbox in frame_result.boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = bbox.xyxy[0].tolist()

            # Crop the image using the bounding box coordinates
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_image_array = np.array(cropped_image)
            cropped_image.show()

            #check if the frame acceptable or not
            valid =Frame_Validation(cropped_image_array, Valid_model)

            prediction_results = prediction_model(cropped_image_array)
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
            
                    result.show()
                    predicted_label += detected_number
            print(f"Processed {image_path}: {predicted_label}") 
            print(f"Frame Validation: {valid}")        
    return predicted_label

input= "D:/HUST/dev/py/Digit_catkhung/data/final/images/28886691_1.png"

frame_model_path = "D:/HUST/dev/py/Digit_catkhung/model/khung/khung_best_1306.pt"
prediction_model_path = "D:/HUST/dev/py/Digit_catkhung/model/so/best_01_so.pt"
Valid_model = "D:/HUST/dev/py/Digit_catkhung/model/khung hop le/best_2107.pt"
predicted_labels = predict_image(input, frame_model_path, prediction_model_path,Valid_model)
print(predicted_labels)
