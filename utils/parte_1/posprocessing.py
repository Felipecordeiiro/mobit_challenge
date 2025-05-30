import cv2 as cv

def suavizacao_bordas(contours, contours_otsu, contours_adapt):
    # Suavizacao nos contornos
    contours = [cnt for cnt in contours if cv.contourArea(cnt) > 100]
    contours_otsu = [cnt for cnt in contours_otsu if cv.contourArea(cnt) > 100]
    contours_adapt = [cnt for cnt in contours_adapt if cv.contourArea(cnt) > 100]

    return contours, contours_otsu, contours_adapt