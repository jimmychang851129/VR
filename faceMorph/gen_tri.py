import cv2
import numpy as np
import random
import dlib
import imutils

def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates


img = cv2.imread("test.jpg");
size = img.shape
rect = (0, 0, size[1], size[0])

subdiv  = cv2.Subdiv2D(rect); 

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

image = imutils.resize(img, width=800)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rets = detector(gray, 1)
# loop over the face detections
point = []
for (i, ret) in enumerate(rets):
	# determine the facial landmarks for the face region, then
	# convert the landmark (x, y)-coordinates to a NumPy array
	shape = predictor(gray, ret)
	shape = shape_to_numpy_array(shape)
	
	for (x, y) in shape:
		#cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		subdiv.insert((x,y))
		point.append((x,y))

subdiv.insert((0,0))
point.append((0,0))
subdiv.insert((size[1]/2,0))
point.append((size[1]/2,0))
subdiv.insert((size[1]-1,0))
point.append((size[1]-1,0))
subdiv.insert((0, size[0]/2))
point.append((0, size[0]/2))
subdiv.insert((0, size[0]-1))
point.append((0, size[0]-1))
subdiv.insert((size[1]/2, size[0]-1))
point.append((size[1]/2, size[0]-1))
subdiv.insert((size[1]-1,size[0]/2))
point.append((size[1]-1,size[0]/2))
subdiv.insert((size[1]-1,size[0]-1))
point.append((size[1]-1,size[0]-1))

List = subdiv.getTriangleList()
tri_coord = []
for ele in List:
	tmp1 = -1
	tmp2 = -1
	tmp3 = -1
	for i in range(76):
		if point[i][0] == ele[0] and point[i][1] == ele[1]:
			tmp1 = i
		if point[i][0] == ele[2] and point[i][1] == ele[3]:
			tmp2 = i
		if point[i][0] == ele[4] and point[i][1] == ele[5]:
			tmp3 = i
	print tmp1, tmp2, tmp3
