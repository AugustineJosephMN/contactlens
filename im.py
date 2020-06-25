# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 10:53:05 2018

@author: Raj Shah
"""
#
import glob
import cv2
from PIL import Image
import os
import sys
import dlib
from skimage import io
from imutils import face_utils
import pandas as pd
import numpy as np
#################################With Video#######################################

imgEye = cv2.imread("/home/senscript/Music/Face-Overlay-AR-master/eye3.png",-1)
cv2.imshow("eye",imgEye)

orig_mask = imgEye[:,:,3]
cv2.imshow("orig_mask", orig_mask)

orig_mask_inv = cv2.bitwise_not(orig_mask)


imgEye = imgEye[:,:,0:3]

predictor = dlib.shape_predictor("predictor_eye_combine_landmarks.dat")
detector = dlib.simple_object_detector("detector_eye.svm1")
detector2 = dlib.get_frontal_face_detector()

# win_det = dlib.image_window()
# win_det.set_image(detector)
## Now let's run the detector and shape_predictor over the images in the faces
## folder and display the results.
# cap = cv2.VideoCapture("/home/ananthu/Downloads/outpy.avi")
frame = cv2.imread("/home/senscript/Desktop/FaceSwap_last/face_swapping/hh.jpg")
frame1=frame.copy()

# im_pil = Image.fromarray(cap)
i = 0

# ret, frame = cap.read()
# print(frame)
# frame = cv2.imread("/home/ananthu/Desktop/Amar.jpg")
dets2 = detector2(frame)
dets = detector(frame)
print("*******", dets)
# print("Number of pair of eyes detected: {}".format(len(dets)))
# print("Number of faces detected: {}".format(len(dets2)))


def place_eye(crop_img1, crop_img2):
    # eyeOverlayHeight=14
    # eyeOverlayWidth=14
    eyeOverlayHeight, eyeOverlayWidth, channels = crop_img1.shape
    # print("sdfsadfadf",eyeOverlayHeight,eyeOverlayWidth)

    # h, w, c = imgEye.shape[:3]
    # print("**", h, w, c)
    eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
    # cv2.imshow("eyeOverlay", eyeOverlay)
    # print("######", eyeOverlay.shape)

    mask = cv2.resize(orig_mask, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    roi = frame[x1:x2, y1:y2]
    face_part = (roi * (1 / 255.0)) * (mask_inv * (1 / 255.0))
    overlay_part = (eyeOverlay * (1 / 255.0)) * (mask * (1 / 255.0))
    cv2.imshow("roi", mask)

    height, width, channel = eyeOverlay.shape[:3]
    # print("roi shape :::: ", roi.shape)
    # roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    cv2.imwrite("mask.jpg",roi)
    cv2.imshow("mask", roi)
    # print("roi_bg shape :::: ", roi_bg.shape)

    # roi_fg= cv2.bitwise_and(eyeOverlay, eyeOverlay, mask=mask)

    # roi_fg = cv2.fastNlMeansDenoisingColored(roi_fg1, None, 20, 20, 10, 21)
    # cv2.imshow("fg", roi_fg)


    dst=cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0)


    # cv2.imshow("dst", dst)

    frame[(shape[0][1] + 3) - int(0.1 * einter):(shape[0][1] + 3) + int(0.1 * einter),
    shape[0][0] - int(0.1 * einter):shape[0][0] + int(0.1 * einter)] = dst

    ################################################################################################################

    eyeOverlayHeight2, eyeOverlayWidth2, channels2 = crop_img2.shape
    # print("sdfsadfadf", eyeOverlayHeight, eyeOverlayWidth)

    # h, w, c = imgEye.shape[:3]
    # print("**", h, w, c)
    eyeOverlay2 = cv2.resize(imgEye, (eyeOverlayWidth2, eyeOverlayHeight2), interpolation=cv2.INTER_AREA)
    # cv2.imshow("eyeOverlay", eyeOverlay)
    # print("######", eyeOverlay.shape)

    mask2 = cv2.resize(orig_mask, (eyeOverlayWidth2, eyeOverlayHeight2), interpolation=cv2.INTER_AREA)
    mask_inv2 = cv2.resize(orig_mask_inv, (eyeOverlayWidth2, eyeOverlayHeight2), interpolation=cv2.INTER_AREA)
    mask_inv2 = cv2.cvtColor(mask_inv2, cv2.COLOR_GRAY2BGR)
    mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)
    roi2 = frame[xx1:xx2, yy1:yy2]
    cv2.imshow("roi", roi)
    face_part2 = (roi2 * (1 / 255.0)) * (mask_inv2 * (1 / 255.0))
    overlay_part2 = (eyeOverlay2 * (1 / 255.0)) * (mask2 * (1 / 255.0))
    height, width, channel = eyeOverlay.shape[:3]
    # print("roi shape :::: ", roi.shape)
    # roi_bg2 = cv2.bitwise_and(roi2, roi2, mask=mask_inv2)
    cv2.imshow("mask", face_part2)
    # print("roi_bg shape :::: ", roi_bg.shape)
    # roi_fg2 = cv2.add(roi_fg2t, np.array([30.0]))
    # roi_fg = cv2.bitwise_and(eyeOverlay2, eyeOverlay2, mask=mask2)


    # cv2.imshow("fg", roi_fg)

    dst2 = cv2.addWeighted(face_part2, 255.0, overlay_part2, 255.0, 0.0)
    cv2.imshow("dst", overlay_part2)
    #
    frame[(shape[1][1] + 3) - int(0.1 * einter):(shape[1][1] + 3) + int(0.1 * einter),
    shape[1][0] - int(0.1 * einter):shape[1][0] + int(0.1 * einter)] = dst2
    return frame




if (dets):
    for k, d in enumerate(dets):
        print("kd", k, d)
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(),
        #                                                                    d.bottom()))
        shape = predictor(frame, d)
        # print("Shape : ", shape)
        # print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
        shape = face_utils.shape_to_np(shape)
        # print("Shapeeee : ", shape)

    # cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),(255,0,0),1)
    einter = np.sqrt((shape[0][0] - shape[1][0]) ** 2 + (shape[0][1] - shape[1][1]) ** 2)
    # print("einter : ", einter)
    # cv2.circle(frame, (shape[0][0], shape[0][1] + 3), 1, (255, 0, 0), -1)
    # cv2.circle(frame, (shape[0][0], shape[0][1] + 3), int(0.1 * einter), (0, 0, 255))
    # print((shape[0][0], shape[0][1] + 3))
    # cv2.imshow("",frame[(shape[0][1] + 3)-int(0.1 * einter):(shape[0][1] + 3)+int(0.1 * einter), shape[0][0]-int(0.1 * einter):shape[0][0]+int(0.1 * einter)])
    # cv2.circle(frame, (shape[1][0], shape[1][1] + 3), 1, (255, 0, 0), -1)
    # cv2.circle(frame, (shape[1][0], shape[1][1] + 3), int(0.1 * einter), (0, 0, 255))
    # print(0.1 * einter)
    crop_img1 = frame[(shape[0][1] + 3) - int(0.1 * einter):(shape[0][1] + 3) + int(0.1 * einter),
                shape[0][0] - int(0.1 * einter):shape[0][0] + int(0.1 * einter)]

    x1 = (shape[0][1] + 3) - int(0.1 * einter)

    x2 = (shape[0][1] + 3) + int(0.1 * einter)
    y1 = shape[0][0] - int(0.1 * einter)
    y2 = shape[0][0] + int(0.1 * einter)

    crop_img2 = frame[(shape[1][1] + 3) - int(0.1 * einter):(shape[1][1] + 3) + int(0.1 * einter),
                shape[1][0] - int(0.1 * einter):shape[1][0] + int(0.1 * einter)]

    xx1 = (shape[1][1] + 3) - int(0.1 * einter)

    xx2 = (shape[1][1] + 3) + int(0.1 * einter)
    yy1 = shape[1][0] - int(0.1 * einter)
    yy2 = shape[1][0] + int(0.1 * einter)
    # cv2.imshow("cropped", crop_img1)
    # cv2.waitKey(0)


    abc = place_eye(crop_img1, crop_img2)
    img_gray1 = cv2.cvtColor(abc, cv2.COLOR_BGR2GRAY)
    mask1 = np.zeros_like(img_gray1)
    # detector1 = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(
        "/home/senscript/Desktop/FaceSwap_last/face_swapping/shape_predictor_81_face_landmarks.dat")
    faces1 = detector2(img_gray1)

    index_dst = []
    for face1 in faces1:
        landmarks1 = predictor(img_gray1, face1)
        landmarks_points1 = []
        for n1 in range(36, 42):
            x = landmarks1.part(n1).x
            y = landmarks1.part(n1).y
            landmarks_points1.append((x, y))
            np.array(landmarks_points1)
            face_points1 = []
            points1 = np.array(landmarks_points1, np.int32)

        #
        convexhull1 = cv2.convexHull(points1)
        # cv2.polylines(img, [points], True, (255, 0, 0), 3)
        cv2.fillConvexPoly(mask1, convexhull1, 255)
        fg = cv2.bitwise_or(abc, abc, mask=mask1)
        mask_inv = cv2.bitwise_not(mask1)
        bk = cv2.bitwise_or(frame1, frame1, mask=mask_inv)
        final = cv2.bitwise_or(fg, bk)
        cv2.imshow("out.jpg", final)
        cv2.waitKey(0)



        # cv2.imshow("fra", frame)

    # cv2.imwrite("out.jpg", abc)
    # cv2.imshow("frame", abc)
    # cv2.imwrite("frame.jpg",abc)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# cap.release()

# cv2.destroyAllWindows()


