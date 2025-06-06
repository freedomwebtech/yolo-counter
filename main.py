import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
names = model.names

# Debug mouse position
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)

cap=cv2.VideoCapture("pe.mov")


frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020,500))
    results = model.track(frame, persist=True,classes=[0,2])

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2
            label = names[class_id]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1 + 3, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
  
    cv2.imshow("RGB", frame)
    
    if cv2.waitKey(0) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

