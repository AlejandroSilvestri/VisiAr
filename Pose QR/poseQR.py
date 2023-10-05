# Detecta código QR, determina la pose del marcador y la expresa como Tcq, tvec & rvec.
# Muestra el contorno del marcador y tres ejes en el origen.
# La variable squareLength establece la longitud de un lado, 
# e indirectamente establece la unidad de medida del mundo virtual 3D. 

import cv2 as cv
import numpy as np

cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Se acaba de ajustar la resolución a 640x480
# Si bien ya se conocen alto y ancho, el siguiente código los obtiene midiendo una imagen
ret = False
while not ret:
    ret, im = cam.read()
ancho, alto, canales = im.shape
print ("Resolución: %d x %d" % (ancho, alto))

distCoeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
cx = ancho / 2 - 0.5
cy = alto / 2 - 0.5
f = 2.4 * cx    # Para una cámara de 57º de apertura visual
cameraMatrix = np.array(
    [[f, 0.0, cx],
     [0.0, f, cy],
     [0.0, 0.0, 1.0]], np.float32)
squareLength = 10 # cm, lado del QR
objectPoints = np.array(
               [[-squareLength / 2,  squareLength / 2, 0],
                [ squareLength / 2,  squareLength / 2, 0],
                [ squareLength / 2, -squareLength / 2, 0],
                [-squareLength / 2, -squareLength / 2, 0]])

det = cv.QRCodeDetector()

# Transformación
Tcq = []

print("ESC to quit.")
countPrint = 0
while True:
    ret, im = cam.read()
    ret, qrs = det.detect(im)   # qr único, no es multi
    countPrint += 1
    if ret:
        imPoints = np.array(qrs, np.float32).reshape((4, 2))
        pts = imPoints.astype(np.int32).reshape((1, 4, 2))
        im = cv.polylines(im, pts, True, color=(0,200,255), thickness=2)

        #ret, rvec, tvec = cv.solvePnP(objectPoints, imPoints, cameraMatrix, distCoeffs, flags=cv.SOLVEPNP_SQPNP)
        ret, rvec, tvec = cv.solvePnP(objectPoints, imPoints, cameraMatrix, distCoeffs, flags=cv.SOLVEPNP_IPPE_SQUARE)

        if ret:
            axis = np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)
            imgpts, _ = cv.projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs)

            imgpts = np.int32(imgpts).reshape(-1,2)
            origin = tuple(imgpts[0].ravel())
            im = cv.line(im, origin, tuple(imgpts[1].ravel()), (255,0,0), 2)
            im = cv.line(im, origin, tuple(imgpts[2].ravel()), (0,255,0), 2)
            im = cv.line(im, origin, tuple(imgpts[3].ravel()), (0,0,255), 2)

            R, _ = cv.Rodrigues(rvec)
            Tcq = np.identity(4)
            Tcq[:3, :3] = R
            Tcq[:3, 3] = tvec.ravel()

            if countPrint>15:
                countPrint = 0
                print("t", tvec.astype(np.int32).flatten())
                print("ang", np.linalg.norm(rvec))
                print(rvec)
                print("Tcq", Tcq)

    cv.imshow("webcam", im)

    match cv.waitKey(30):
        case 27: # ESC
            break
