import cv2 as cv
from ultralytics import YOLO
from utils.parte_2.views import view, view_results

# Versão nano
model_n = YOLO("yolov8n.pt")
# Versão small
model_s = YOLO("yolov8s.pt")

img = cv.imread("data/person.png")
results = model_n.predict(source=img, save=True, project='./results/parte_2/n/')
results_2 = model_s.predict(source=img, save=True, project='./results/parte_2/s/')


person_count = view_results(results)
person_count_2 = view_results(results_2)
print(f"Pessoas detectadas (usando yolov8n): {person_count}")
print(f"Pessoas detectadas (usando yolov8s): {person_count_2}")

path_image = "./results/parte_2/n/predict/image0.jpg"
path_image_2 = "./results/parte_2/s/predict/image0.jpg"
view(path_image, path_image_2)