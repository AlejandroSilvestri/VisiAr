'''
Genera un render de un modelo 3D en una pose fija, y lo mezcla con el video de la webcam
'''

import numpy as np
import cv2 as cv
import trimesh
import pyrender

triMesh = trimesh.load('fuze.obj')
mesh = pyrender.Mesh.from_trimesh(triMesh)
scene = pyrender.Scene()
scene.add(mesh)
#pyrender.Viewer(scene, use_raymond_lighting=True)

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=4.0/3.0)
s = np.sqrt(2)/2
camera_pose = np.array([
    [1.0,  0.0, 0.0, 0.0],
    [0.0, -s,   s,   0.3],
    [0.0,  s,   s,   0.35],
    [0.0,  0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)
light = pyrender.SpotLight(color=np.ones(3), 
                           intensity=3.0,
                           innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose)
r = pyrender.OffscreenRenderer(640, 480)

colorRGB, depth = r.render(scene)
colorBGR = cv.cvtColor(colorRGB, cv.COLOR_RGB2BGR)

# Perform alpha blending for BGR channels
# model: BGR image with model
# depth: gray image, 0 means background, non 0 means model
# background: BGR image, usually from live camera
def blend(model, depth, background):
    return np.where(depth[:, :, np.newaxis] != 0, model[:, :, :3], background[:, :, :3])


cam = cv.VideoCapture(0)
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ret, im = cam.read()
    cv.imshow('Realidad aumentada', blend(colorBGR, depth, im))

    match cv.waitKey(30):
        case 27: # ESC
            break
    