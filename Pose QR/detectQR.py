# Ejemplo de detección de QR
# Dibuja el contorno del QR, muestra el texto y enumera vértices
# La resolución de la cámara está reducida

import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)
ancho = cam.get(cv.CAP_PROP_FRAME_WIDTH)
alto = cam.get(cv.CAP_PROP_FRAME_HEIGHT)
print ("Resolución máxima: %d x %d" % (ancho, alto))
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

det = cv.QRCodeDetector()

print("ESC to quit.")
while True:
    ret, im = cam.read()
    #ret, qrs = det.detect(im)   # detecta sin decodificar; ret es booleano
    code, qrs, straight_qrcode = det.detectAndDecode(im)   # detecta y decodifica, code es la string decodificada
    if qrs is not None:
        # qr shape: (1,4,2), y si fuera multi: (n, 4, 2)
        # n o 1: cantidad de QR
        # 4: esquinas por cada QR
        # 2: coordenadas (x,y) de cada esquina
        pts = np.array(qrs, np.int32).reshape((-1, 4, 2))   # aquí reshape puede ser redundante
        im = cv.polylines(im, pts, True, color=(0,200,255), thickness=2)
        im = cv.putText(im, "1", pts[0,1], cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0))
        im = cv.putText(im, "2", pts[0,2], cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0))
        im = cv.putText(im, "3", pts[0,3], cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0))
        if code:
            im = cv.putText(im, code, pts[0,0], cv.FONT_HERSHEY_SIMPLEX, 0.5, (64,64,0))
        else:
            im = cv.putText(im, "0", pts[0,0], cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,128,0))

        if straight_qrcode is not None:
                cv.imshow("QR", straight_qrcode)

    cv.imshow("webcam", im)

    match cv.waitKey(30):
        case 27: # ESC
            break
