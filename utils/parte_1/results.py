import cv2 as cv

def save_result(image, filename):
    filepath = f"./results/parte_1/{filename}.png"
    cv.imwrite(filepath, image)
    print(f"Image salva em: {filepath}")