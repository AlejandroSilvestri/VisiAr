# Some QR detection demos

These demos work on any QR code, two QR images are provided for convenience.
All demos work over webcam, ESC exists the loop and quits.

## detectQR.py
 Detects and outlines a single QR in an image, enumerates its four vertices and displays the string encoded in the QR.

## poseQR.py
Also detects and outlines a single QR in an image, and:

* computes the QR pose in 3D space
* draws three coordinates axes on QR
* composes and shows in console the 4x4 pose matrix Tcq of the QR code in the camera reference system.

poseQR.py requires a calibrated camera, meaning you must insert in the code the appropiated intrinsics and distortion coefficients values.