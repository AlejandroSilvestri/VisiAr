'''
Genera un render del modelo, y lo muestra junto a su mapa de profundidad
'''
import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt

triMesh = trimesh.load('modelos/fuze.obj')
mesh = pyrender.Mesh.from_trimesh(triMesh)
scene = pyrender.Scene()
scene.add(mesh)
#pyrender.Viewer(scene, use_raymond_lighting=True)

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=4.0/3.0)
s = np.sqrt(2)/2
camera_pose1 = np.array([
    [0.0, 0.0, 1.0, 0.50],
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])
camera_pose2 = np.array([
    [1.0, 0.0, 0.0, 0.10],
    [0.0,  -s,   s, 0.35],
    [0.0,   s,   s, 0.35],
    [0.0, 0.0, 0.0, 1.0],
])
camera_pose3 = np.array([
    [0.0, -s,   s,   0.25],
    [1.0,  0.0, 0.0, 0.0],
    [0.0,  s,   s,   0.35],
    [0.0,  0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose1)
light = pyrender.SpotLight(color=np.ones(3), 
                           intensity=3.0,
                           innerConeAngle=np.pi/16.0,
                           outerConeAngle=np.pi/6.0)
scene.add(light, pose=camera_pose1)

# offscreen render
r = pyrender.OffscreenRenderer(640, 480)
colorRGB, inv_depth = r.render(scene)

# show render and inverse depth
plt.figure()
plt.subplot(1,2,1)
plt.axis('off')
plt.imshow(colorRGB)
plt.subplot(1,2,2)
plt.axis('off')
plt.imshow(inv_depth)
plt.show()
