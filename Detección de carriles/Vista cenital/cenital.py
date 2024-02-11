import numpy as np
import cv2 as cv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help="flujo de video de entrada: la ruta de un archivo de video o el número de cámara", default=0)
parser.add_argument("-l", "--load", help="archivo donde cargar los resultados")
parser.add_argument("-s", "--save", help="archivo donde guardar los resultados")
args = parser.parse_args()

if(not args.save or not args.load):
    filename = args.video + '.yaml'
    if(not args.save):
        args.save = filename
    if(not args.load):
        args.load = filename

# Util
play = True # bandera de control play/pausa
fs = None   # yaml



# Video
video = cv.VideoCapture(args.video)
anchoImagen = video.get(cv.CAP_PROP_FRAME_WIDTH)
altoImagen = video.get(cv.CAP_PROP_FRAME_HEIGHT)
print('Resolución del video:', anchoImagen, altoImagen)

alturaObjetivo = 480
cenitalLado = 500
cenitalTamano = (cenitalLado,cenitalLado)


if(altoImagen > alturaObjetivo):
    anchoObjetivo = int(anchoImagen/altoImagen * alturaObjetivo)
    video.set(cv.CAP_PROP_FRAME_WIDTH,  anchoObjetivo)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, alturaObjetivo)

_, im = video.read()
(altoImagen, anchoImagen, canales) = im.shape
print('Resolución ajustada:', anchoImagen, altoImagen, type(anchoImagen))
tamanoObjetivo = None
if(altoImagen > alturaObjetivo):
    altoImagen = alturaObjetivo
    anchoImagen = anchoObjetivo
    tamanoObjetivo = (int(anchoImagen), int(altoImagen))




# GUI
def mouse(event,x,y,flags,param):
    global comando
    if(event == cv.EVENT_LBUTTONDOWN):
        if(comando == Comando.HORIZONTAL):
            global horizonte
            horizonte = int(y)
            calcularRoi()
            #comando = ''
        elif(comando == Comando.TOPE):
            global tope
            tope = int(y)
            calcularRoi()
            #comando = ''

cv.namedWindow('video')
cv.setMouseCallback('video', mouse)

class Comando:
    HORIZONTAL = 'h'
    TOPE = 't'
    VACIO = ''

comando = ''

horizonte = int(altoImagen * 0.5)
tope = int(altoImagen * 0.75)

def dibujarLineaHorizontal(im, y, texto, seleccionada=False):
    y = int(y)
    cv.putText(im, texto, (0, y), cv.FONT_HERSHEY_PLAIN, 1.0, (255,255,255))
    if seleccionada:
        # Línea seleccionada
        cv.line(imAnotada, (0, y), (anchoImagen, y), (255,255,255), 2)
    else:
        cv.line(imAnotada, (0, y), (anchoImagen, y), (128,128,128))

    return

# ROI
# Cuadrilátero, comenzando por vértice superior izquierdo, en sentido horario
# Si se proporciona una homografía, la usa para obtener horizonte y tope
def calcularRoi(H_ = None):
    global roiPoly, fuga, H

    if(H_ is not None):
        # Homografía suministrada, se recalculan horizonte y tope
        global horizonte, tope
        H = H_.astype(np.float32)
        puntosClave = np.array(((0,-1,0),(0,0,1)), np.float32)
        print('puntosClave', puntosClave)
        puntosEnPerspectiva = np.matmul(np.linalg.inv(H.astype(np.float32)), puntosClave.T).T
        print('puntosEnPerspectiva', puntosEnPerspectiva)
        horizonte = int(puntosEnPerspectiva[0,1]/puntosEnPerspectiva[0,2])
        tope      = int(puntosEnPerspectiva[1,1]/puntosEnPerspectiva[1,2])
        print('horizonte y tope', horizonte, tope)

    medio = int(anchoImagen/2)
    fuga = (medio, horizonte)
    xProyectado = int(medio*(altoImagen-tope)/(altoImagen-horizonte))
    roiVertices = np.array([
        [xProyectado,tope],
        [anchoImagen-xProyectado,tope],
        [anchoImagen,altoImagen],
        [0,altoImagen]
    ], np.int32)
    roiPoly = roiVertices.reshape((-1,1,2))

    cenitalVertices = np.array([
        [0,0],
        [cenitalLado,0],
        [cenitalLado,cenitalLado],
        [0,cenitalLado]
    ], np.float32)

    if(H_ is None):
        H = cv.getPerspectiveTransform(roiVertices.astype(np.float32), cenitalVertices)
        print('Homografía:\n', H, '\n')

calcularRoi()

umbralCanny = 160
def umbralCannyTrackbar(valor):
    global umbralCanny, detector
    umbralCanny = valor
    detector = cv.ximgproc.createFastLineDetector(canny_th1 = umbralCanny, canny_th2 = umbralCanny*3)


cv.namedWindow('Canny')
cv.createTrackbar('umbral', 'Canny' , 0, 255, umbralCannyTrackbar)

umbralBinario = 160
def umbralBinarioTrackbar(valor):
    global umbralBinario
    umbralBinario = valor

cv.namedWindow('Binario')
cv.createTrackbar('umbral', 'Binario' , 0, 255, umbralBinarioTrackbar)


# Carga la matriz desde yaml, y actualiza la anotación.  El nombre del archivo está en args.load. Si el archivo no existe no hace nada.
def load():
    fs = cv.FileStorage(args.load, cv.FILE_STORAGE_READ)
    if(fs.isOpened()):
        global H
        H = fs.getNode('H').mat()
        calcularRoi(H)
        fs.release()

# Guarda la matriz en yaml.  El nombre del archivo está en args.save.  Si el archivo no existe lo crea.
def save():
    fs = cv.FileStorage(args.save, cv.FILE_STORAGE_WRITE)
    fs.write('H', H)
    fs.release()


load()

detector = cv.ximgproc.createFastLineDetector(canny_th1 = umbralCanny, canny_th2 = umbralCanny*3)
#cenitalImAnterior = np.zeros((cenitalLado, cenitalLado), np.float32)
while(True):
    if(play):
        _, im = video.read()
        if(tamanoObjetivo):
            im = cv.resize(im, tamanoObjetivo)

    imGris = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # Vista cenital
    cenitalIm = cv.warpPerspective(im, H, cenitalTamano)
    # Phase correlation: la escena no sirve para medir el movimiento sobre la vista cenital; es posible que tampoco funcione optical flow sobre cenital.
    #cenitalImGris = cv.cvtColor(cenitalIm, cv.COLOR_BGR2GRAY).astype(np.float32)
    #direccion = cv.phaseCorrelate(cenitalImAnterior, cenitalImGris)
    #cenitalImAnterior = cenitalImGris
    #print('direccion', direccion)
    cv.imshow('cenital', cenitalIm)


    # Fast Line Detector
    lineas = detector.detect(imGris)
    
    # Anotaciones
    imAnotada = im.copy()
    dibujarLineaHorizontal(imAnotada, horizonte, 'horizonte', comando == Comando.HORIZONTAL)
    dibujarLineaHorizontal(imAnotada, tope, 'tope', comando == Comando.TOPE)
    cv.line(imAnotada, roiPoly[3,0], fuga, (128,128,128))
    cv.line(imAnotada, roiPoly[2,0], fuga, (128,128,128))
    cv.polylines(imAnotada, [roiPoly], True, (0,255,0))
    cv.imshow('video', imAnotada)

    # Canny
    imCanny = cv.Canny(imGris, umbralCanny, umbralCanny*3)
    cv.imshow('Canny', imCanny)

    # Binario
    imBinaria = cv.inRange(imGris, umbralBinario, 256)
    imBinaria = detector.drawSegments(imBinaria, lineas)
    cv.imshow('Binario', imBinaria)

    tecla = cv.waitKey(30)



    match tecla:
        case -1:
            continue
        case 27:
            break

    match chr(tecla):
        case ' ':
            play = not play
        case 'h':
            # Ajustar horizonte
            if(comando == Comando.HORIZONTAL):
                comando = ''
            else:
                comando = Comando.HORIZONTAL
        case 't':
            # Ajustar límite
            if(comando == Comando.TOPE):
                comando = ''
            else:
                comando = Comando.TOPE
        case 'p':
            print(roiPoly, type(roiPoly), type(roiPoly[0]), type(roiPoly[0][0]))
        
        case 's':
            save()
        case 'l':
            load()


if(fs):
    fs.release()

cv.destroyAllWindows()