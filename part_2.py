import cv2 as cv
from ultralytics import YOLO

# Versão nano
model = YOLO("yolov8n.pt")

img = cv.imread("data/person.png")
results = model.predict(source=img, save=True)

for r in results:
  print("Resultados importantes:")
  print(r.boxes.data)
  print(r.boxes.cls)
  print(r.boxes.xyxy)
  classes = r.boxes.cls.cpu().numpy()
  person_count = (classes == 0).sum()  # Baseado na anotação do dataset COCO, a classe 0 é person
  print(f"Pessoas detectadas: {person_count}")
