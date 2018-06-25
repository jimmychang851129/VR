#!/usr/bin/env python

import numpy as np
import cv2
import sys
import dlib
import imutils

def append_new_point(size, point):
    point.append((0,0))
    point.append((size[1]/2,0))
    point.append((size[1]-1,0))
    point.append((0, size[0]/2))
    point.append((0, size[0]-16))
    point.append((size[1]/2, size[0]-16))
    point.append((size[1]-1,size[0]/2))
    point.append((size[1],size[0]-16))
    '''
    for i in range(3):
        point.append((0,0))
    '''
def shape_to_numpy_array(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coordinates = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coordinates[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coordinates

# Read points from text file
def readPoints(path) :
    # Create an array of points.
    points = [];
    # Read points
    with open(path) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    return points

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2
    #print(imgRect.shape)
    #print(img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]].shape)

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


if __name__ == '__main__' :
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../shape_predictor_68_face_landmarks.dat")

    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    alpha = 0.2
    
    # Read images
    img1 = cv2.imread(filename1);
    img2 = cv2.imread(filename2);
    #size1 = img1.shape
    #img2 = imutils.resize(img2, width=size1[1], height=size1[0])

    # get points by dlib
    points1 = []
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    rets = detector(gray, 1)
    for (i, ret) in enumerate(rets):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, ret)
        shape = shape_to_numpy_array(shape)
        for (x, y) in shape:
            points1.append((x,y))
        break
    append_new_point(img1.shape, points1)

    # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # Read array of corresponding points
    points2 = readPoints(filename2 + '.txt')
    points = [];

    # Compute weighted average point coordinates
    for i in range(0, len(points1)):
        x = ( 1 - alpha ) * points1[i][0] + alpha * points2[i][0]
        y = ( 1 - alpha ) * points1[i][1] + alpha * points2[i][1]
        points.append((x,y))


    # Allocate space for final output
    imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

    # Read triangles from tri.txt
    with open("tri76.txt") as file :
        for line in file :
            x,y,z = line.split()
            
            x = int(x)
            y = int(y)
            z = int(z)
            
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [ points[x], points[y], points[z] ]

            # Morph one triangle at a time.
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)
    '''
    tmp_size = imgMorph.shape
    for i in range(tmp_size[0]):
        for j in range(tmp_size[1]):
            if imgMorph[i][j][0] == 0 and imgMorph[i][j][1] == 0 and imgMorph[i][j][2] == 0:
                imgMorph[i][j] = img1[i][j]
    '''
    #Display Result
    #cv2.imshow("Morphed Face", np.uint8(imgMorph))
    cv2.imwrite("output.jpg", np.uint8(imgMorph))
    #cv2.waitKey(50000)
