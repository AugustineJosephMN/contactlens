import sys
import base64
from test import YOLO, detect_video
from PIL import Image
import cv2, numpy
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


class yoloCamera():
    def __init__(self, cameraId=0):
        self.cameraId = cameraId
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)
        self.yolocore = YOLO()
        # self.camera = cv2.VideoCapture(0)

    def getFrame(self,img):
        # if not self.camera.isOpened():
        #     return b''
        # ret, img = self.camera.read()
        # img_pil = Image.fromarray(img)

        # ---------------------------------------------
        r_img = detect_video(self.yolocore, img)

        image = cv2.imencode('.jpg', r_img)[1]
        return base64.b64encode(image).decode('utf-8', 'strict')


        # if r_img is None:
        #     ret1, jpeg = cv2.imencode('.jpg', img)
        # # r_img_cv = numpy.array(r_img)
        # else:
        #     ret1, jpeg = cv2.imencode('.jpg', r_img)
        # return jpeg.tobytes()

    def close(self):
        self.yolocore.close_session()

# from flask import jsonify

class InvalidException(Exception):
    status_code = 500

    def __init__(self, message, status_code=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code

    def __str__(self):
        return self.message

