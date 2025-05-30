import cv2 as cv

def blurring_image(image):
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    blurred_image = cv.GaussianBlur(gray_image, (7, 7), 0)

    return blurred_image