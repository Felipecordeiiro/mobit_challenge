import cv2 as cv
import matplotlib.pyplot as plt

def download_comparative(filename, image_rgb, binary_image, binary_image_otsu, binary_image_adap):
    plt.figure(figsize=(16, 4))
    titles = ['Original', 'Threshold Fixo', 'Otsu', 'Adaptativo']

    images = [image_rgb, binary_image, binary_image_otsu, binary_image_adap]

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        # Para imagens binárias, use cmap='gray'
        if i == 0:
            plt.imshow(images[i])
        else:
            plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f"./results/parte_1/{filename}.png", bbox_inches='tight', dpi=300)
    plt.show()
    

def draw_contours(image, contours, contours_otsu, contours_adapt):
    # Visualiza os contornos na imagem original
    img_contornos = image.copy()
    img_contornos_2 = image.copy()
    img_contornos_3 = image.copy()

    cv.drawContours(img_contornos, contours, -1, (255, 0, 0), 2)
    cv.drawContours(img_contornos_2, contours_otsu, -1, (255, 0, 0), 2)
    cv.drawContours(img_contornos_3, contours_adapt, -1, (255, 0, 0), 2)
    
    return img_contornos, img_contornos_2, img_contornos_3