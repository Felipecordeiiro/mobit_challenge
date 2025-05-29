import cv2 as cv
import matplotlib.pyplot as plt

def view(image, num_objetos):
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title(f'Objetos detectados: {num_objetos}')
    plt.axis('off')
    plt.show()

def preprocess(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    blurred_image = cv.GaussianBlur(gray_image, (7, 7), 0)

    return blurred_image

image = cv.imread("data/graos.png")
img_preprocessed = preprocess(image)

# Binarizando a imagem (0 - 1)
_, binary_image = cv.threshold(img_preprocessed, 50, 255, cv.THRESH_BINARY)

contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Pós-processamento suave nos contornos
contours = [cnt for cnt in contours if cv.contourArea(cnt) > 100]

# Conta os objetos
num_objetos = len(contours)
print(f"Número de objetos detectados: {num_objetos}")

# Visualiza os contornos na imagem original
img_contornos = image.copy()
cv.drawContours(img_contornos, contours, -1, (255, 0, 0), 2)

#cv.imwrite("output_image.png", image)
cv.imshow("Imagem com contornos", img_contornos)
cv.waitKey(0)
cv.destroyAllWindows()