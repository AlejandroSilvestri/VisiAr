import numpy as np
import cv2 as cv

im0 = cv.imread('intersection binaria.png', cv.IMREAD_GRAYSCALE)
im0 = cv.resize(im0, [500,500])
im1 = cv.imread('Captura de pantalla de bfmc2020_online_2.avi.png', cv.IMREAD_GRAYSCALE)
im1 = cv.resize(im1, [640,480])
_, im1 = cv.threshold(im1, 136, 255, cv.THRESH_BINARY)
cv.imshow('Patron', im0)
cv.imshow('Perspectiva', im1)

H = np.identity(3, np.float32)
H = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
])
print(H.dtype)

im1p = cv.warpPerspective(im1, H, im1.shape)

cv.waitKey()
resultado = cv.findTransformECC(im0, im1, H, cv.MOTION_HOMOGRAPHY)
print(resultado)

print(H)

im2 = cv.warpPerspective(im0, H, im1.shape)
cv.imshow('Resultado', im2)

cv.waitKey()