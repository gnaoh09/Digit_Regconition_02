from ultralytics import YOLO

# Load the model
model = YOLO('D:/HUST/dev/py/Digit_catkhung/model/khung/khung_best_1306.pt')

results = model("D:/HUST/dev/py/Digit_catkhung/data/final/images/28638981_1.png")

# Display the results with bounding boxes and labels
for result in results:
    result.show()
