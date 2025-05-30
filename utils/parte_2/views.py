import cv2 as cv

def view(path_1, path_2):
  img_pred = cv.imread(path_1)
  img_pred_2 = cv.imread(path_2)
  cv.imshow("Deteccao usando yolov8n", img_pred)
  cv.imshow("Deteccao usando yolov8s", img_pred_2)
  cv.waitKey(0)
  cv.destroyAllWindows()

def view_results(results):
  for r in results:
    print("Classes detectadas:")
    print(r.boxes.cls)
    print("Bouding boxes:")
    print(r.boxes.xyxy)
    classes = r.boxes.cls.cpu().numpy()
    person_count = (classes == 0).sum()  # Baseado na anotação do dataset COCO, a classe 0 é person
    return person_count