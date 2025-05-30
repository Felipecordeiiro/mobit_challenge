import cv2 as cv

def creating_thresholds(img_preprocessed):
    # Tipos de segmentacao analisadas
    _, binary_image = cv.threshold(img_preprocessed, 50, 255, cv.THRESH_BINARY)
    _, binary_image_otsu = cv.threshold(img_preprocessed, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    binary_image_adap = cv.adaptiveThreshold(
        img_preprocessed, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv.THRESH_BINARY, 11, 2)
    
    return binary_image, binary_image_otsu, binary_image_adap

def finding_contours(binary_image, binary_image_otsu, binary_image_adap):

    contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_otsu, _ = cv.findContours(binary_image_otsu, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_adapt, _ = cv.findContours(binary_image_adap, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    return contours, contours_otsu, contours_adapt