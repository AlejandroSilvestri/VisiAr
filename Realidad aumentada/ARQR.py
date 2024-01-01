# Detecta código QR, determina la pose del marcador y la expresa como Tcq, tvec & rvec.
# Genera un render de un modelo con al misma pose que el marcador, y lo superpone a la imagen de la webcam.
# La variable qrGl establece la longitud de un lado del QR, y la unidad de medida del mundo virtual 3D. 

import cv2 as cv
import numpy as np
import trimesh
import pyrender

# Webcam, su flujo de video, sus parámetros de cámara
webcam = cv.VideoCapture(0)
webcam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
webcam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Se acaba de ajustar la resolución a 640x480
# Si bien ya se conocen alto y ancho, el siguiente código los obtiene midiendo una imagen
ret = False
while not ret:
    ret, im = webcam.read()
alto, ancho, canales = im.shape
print ("Resolución: %d x %d" % (ancho, alto))

distorsion = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
cx = ancho / 2 - 0.5
cy = alto / 2 - 0.5
f = 2.4 * cx    # Para una cámara de 55º de apertura visual

K = np.array(
    [[f, 0.0, cx],
     [0.0, f, cy],
     [0.0, 0.0, 1.0]], np.float32)


# Código QR, sus dimensiones virtuales en openGL, terna XYZ para anotar
qrGl = 0.30 # lado del QR en dimensiones virtuales de GL
largoEje = qrGl/3.0
ternaXYZ = np.float32([[0,0,0], [largoEje,0,0], [0,largoEje,0], [0,0,largoEje]]).reshape(-1,3)
modelo3DCuadradoQR = np.array(
               [[-qrGl / 2,  qrGl / 2, 0],
                [ qrGl / 2,  qrGl / 2, 0],
                [ qrGl / 2, -qrGl / 2, 0],
                [-qrGl / 2, -qrGl / 2, 0]])

# Transformaciones
'''
Sistemas de referencias:
q: código QR
c: cámara real, según OpenCV (z hacia adelante,y hacia abajo)
g: cámara virtual, según openGL (z hacia atrás, y hacia arriba)
'''
Tcq = []
Tcg = np.eye(4)
Tcg[1,1] = -1.0
Tcg[2,2] = -1.0


# Modelo
malla = trimesh.load('modelos/fuze.obj')
#malla = trimesh.load('modelos/12221_Cat_v1_l3.obj')    # No funciona
#malla = trimesh.load('modelos/E 45 Aircraft_obj.obj')  # No funciona

scene = pyrender.Scene()
scene.add(pyrender.Mesh.from_trimesh(malla))
#pyrender.Viewer(scene, use_raymond_lighting=True)  # Previsualización del modelo

renderer = pyrender.OffscreenRenderer(ancho, alto)
camaraVirtual = scene.add(pyrender.IntrinsicsCamera(f, f, cx, cy))
iluminacion = scene.add(pyrender.SpotLight(color=np.ones(3), 
                                         intensity=3.0,                           
                                         innerConeAngle=np.pi/16.0,                           
                                         outerConeAngle=np.pi/6.0)
)

# Coloca la imagen del modelo sobre la imagen de fondo
# imagenDelModelo: BGR
# mascaraDelModelo: monocromática, 0 para píxeles de fondo
# imagenDefondo: BGR, típicamente de webcam
def aumentar(imagenDelModelo, mascaraDelModelo, imagenDeFondo):
    return np.where(mascaraDelModelo[:, :, np.newaxis] != 0, imagenDelModelo[:, :, :3], imagenDeFondo[:, :, :3])


detector = cv.QRCodeDetector()
print("ESC to quit.")
while True:
    ret, im = webcam.read()
    ret, qrs = detector.detect(im)   # qr único, no es multi
    if ret:
        verticesQR = np.array(qrs, np.float32).reshape((4, 2))

        # PnP
        ret, rvec, tvec = cv.solvePnP(modelo3DCuadradoQR, verticesQR, K, distorsion, flags=cv.SOLVEPNP_IPPE_SQUARE)

        if False:
            # Recuadro del QR
            pts = verticesQR.astype(np.int32).reshape((1, 4, 2))
            im = cv.polylines(im, pts, True, color=(0,200,255), thickness=2)

        if ret:
            if False:
                # Proyectar ejes
                imgpts, _ = cv.projectPoints(ternaXYZ, rvec, tvec, K, distorsion)

                imgpts = np.int32(imgpts).reshape(-1,2)
                origin = tuple(imgpts[0].ravel())
                im = cv.line(im, origin, tuple(imgpts[1].ravel()), (255,0,0), 2)
                im = cv.line(im, origin, tuple(imgpts[2].ravel()), (0,255,0), 2)
                im = cv.line(im, origin, tuple(imgpts[3].ravel()), (0,0,255), 2)

            # Construir Tcq: pose del QR, matriz de 4x4
            R, _ = cv.Rodrigues(rvec)
            Tcq = np.identity(4)
            Tcq[:3, :3] = R
            Tcq[:3, 3] = tvec.ravel()

            # Pose de cámara respecto de QR
            Tqc = np.linalg.inv(Tcq)
            Tqg = np.matmul(Tqc, Tcg)
            #virtualCameraPose = testPose

            # Proyectar modelo
            scene.set_pose(camaraVirtual, pose=Tqg)
            scene.set_pose(iluminacion,   pose=Tqg)
            
            colorRGB, depth = renderer.render(scene)
            colorBGR = cv.cvtColor(colorRGB, cv.COLOR_RGB2BGR)
            im = aumentar(colorBGR, depth, im)

    cv.imshow("Realidad aumentada", im)

    match cv.waitKey(30):
        case 27: # ESC
            break