import cv2 as cv
from utils.parte_1.posprocessing import suavizacao_bordas
from utils.parte_1.preprocess import blurring_image
from utils.parte_1.process import creating_thresholds, finding_contours
from utils.parte_1.results import save_result
from utils.parte_1.views import download_comparative, draw_contours

# Pre-processamento
image = cv.imread("data/graos.png")

img_preprocessed = blurring_image(image)

# Processamento
binary_image, binary_image_otsu, binary_image_adap = creating_thresholds(img_preprocessed)

image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

download_comparative("comparative_tresholds", image_rgb, binary_image, binary_image_otsu, binary_image_adap)

contours, contours_otsu, contours_adapt = finding_contours(binary_image, binary_image_otsu, binary_image_adap)

# Pós-processamento
contours, contours_otsu, contours_adapt = suavizacao_bordas(contours, contours_otsu, contours_adapt)

# Conta os objetos
num_objetos = len(contours)
print(f"Número de objetos detectados: {num_objetos}")
num_objetos_otsu = len(contours_otsu)
print(f"Número de objetos detectados (usando otsu): {num_objetos_otsu}")
num_objetos_adap = len(contours_adapt)
print(f"Número de objetos detectados (usando adapt): {num_objetos_adap}")

# Visualiza os contornos na imagem original
img_contornos, img_contornos_2, img_contornos_3 = draw_contours(image, contours, contours_otsu, contours_adapt)

cv.putText(img_contornos, f'Objetos: {num_objetos}', (10, 40), 
           cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv.putText(img_contornos_2, f'Objetos: {num_objetos_otsu}', (10, 40), 
           cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
cv.putText(img_contornos_3, f'Objetos: {num_objetos_adap}', (10, 40), 
           cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

save_result(img_contornos, "threshold_fixo")
save_result(img_contornos_2, "threshold_otsu")
save_result(img_contornos_3, "threshold_adaptativo")

download_comparative("comparativo_obj_detectados",image_rgb, img_contornos, img_contornos_2, img_contornos_3)

cv.imshow("Imagem com contornos", img_contornos)
cv.imshow("Imagem com contornos (usando otsu)", img_contornos_2)
cv.imshow("Imagem com contornos (usando adap)", img_contornos_3)
cv.waitKey(0)
cv.destroyAllWindows()