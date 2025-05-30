import cv2 as cv
from ultralytics import YOLO

# Versão nano
model = YOLO("yolov8n.pt")

img = cv.imread("data/person.png")
results = model.predict(source=img, save=True, project='./results/parte_2/')

for r in results:
  print("Resultados importantes:")
  print(r.boxes.data)
  print(r.boxes.cls)
  print(r.boxes.xyxy)
  classes = r.boxes.cls.cpu().numpy()
  person_count = (classes == 0).sum()  # Baseado na anotação do dataset COCO, a classe 0 é person
  print(f"Pessoas detectadas: {person_count}")

def view(path):
  img_pred = cv.imread(path)
  cv.imshow("Deteccao", img_pred)
  cv.waitKey(0)
  cv.destroyAllWindows()

path_image = "./results/parte_2/predict/image0.jpg"
view(path_image)