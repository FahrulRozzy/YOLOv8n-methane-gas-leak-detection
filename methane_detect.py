import cv2
from ultralytics import YOLO

# model = YOLO("C:/Users/fahru/Downloads/SGD128_0001_0-20240609T132708Z-001/SGD128_0001_0/weights/best.pt")
model = YOLO("SGD128_0001_0/weights/best.pt")

video_path = "test-skripsi.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while True:
    # Read a frame from the video
    success, frame = cap.read()
    
    if not success:
        print("Failed to grab frame")
        break
    # Run YOLOv8 inference on the frame
    results = model(frame,max_det=1)
    # Visualize the results on the frame
    annotated_frame = results[0].plot(labels=False, conf=False)
    boxes = results[0].boxes.xywh.cpu()
    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    # Break the loop if 'q' is pressed
    if cv2.waitKey(25) & 0xFF == ord("q"):
        print("Exiting loop")
        break
    

cap.release()
cv2.destroyAllWindows()  