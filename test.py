import cv2
from collections import OrderedDict
import numpy as np
import dlib
import imutils

facial_features_cordinates = {}

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
    ("Mouth", (48, 68)),
    ("Right_Eyebrow", (17, 22)),
    ("Left_Eyebrow", (22, 27)),
    ("Right_Eye", (36, 42)),
    ("Left_Eye", (42, 48)),
    ("Nose", (27, 35)),
    ("Jaw", (0, 17))
])

def AddMask(ret,image,file2):
    left = int(ret.left()) if ret.left() > 0 else 0
    top = int(ret.top()) if ret.top() > 0 else 0
    right = int(ret.right()) if ret.right() < image.shape[1]-1 else image.shape[1]-1
    down = int(ret.bottom()) if ret.bottom() < image.shape[0]-1 else image.shape[0]-1
    show = cv2.imread(file2)
    show = cv2.resize(show,(right+1-left,down+1-top))
    img2gray = cv2.cvtColor(show,cv2.COLOR_BGR2GRAY)
    img2gray = cv2.bitwise_not(img2gray)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    # print("diff = ",top,down,right,left,"mask = ",mask_inv.shape,image[top:down+1,left:right+1].shape)
    img1_bg = cv2.bitwise_and(image[top:down+1,left:right+1],image[top:down+1,left:right+1],mask = mask_inv)
    img2_fg = cv2.bitwise_and(show,show,mask = mask)
    image[top:down+1,left:right+1] = cv2.add(img1_bg,img2_fg)
    return image

def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # create two copies of the input image -- one for the
    # overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    # if the colors list is None, initialize it with a unique
    # color for each facial landmark region
    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
    for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
        # grab the (x, y)-coordinates associated with the
        # face landmark
        (j, k) = FACIAL_LANDMARKS_INDEXES[name]
        pts = shape[j:k]
        facial_features_cordinates[name] = pts

        # check if are supposed to draw the jawline
        if name == "Jaw":
            # since the jawline is a non-enclosed facial region,
            # just draw lines between the (x, y)-coordinates
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        # otherwise, compute the convex hull of the facial
        # landmark coordinates points and display it
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    # return the output image
    # print(facial_features_cordinates)
    return output

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

while rval:
    # image = cv2.resize(frame,(500,500))
    image = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rets = detector(gray, 1)
    # loop over the face detections
    for (i, ret) in enumerate(rets):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, ret)
        shape = shape_to_numpy_array(shape)
        image = AddMask(ret,image,"show.jpg")
        # for (x, y) in shape:
        # 	cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
    # frame = output
    # add
    cv2.imshow("preview", image)
    cv2.waitKey(100)			# latency
    rval, frame = vc.read()
cv2.destroyWindow("preview")

#############
# Reference #
#############
# https://blog.gtwang.org/programming/opencv-webcam-video-capture-and-file-write-tutorial/
# https://www.pyimagesearch.com/2017/04/17/real-time-facial-landmark-detection-opencv-python-dlib/