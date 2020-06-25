# -*- coding: utf-8 -*-

import colorsys
import os
import cv2
from timeit import default_timer as timer

import dlib
import numpy as np
import tensorflow as tf
from keras import backend as K

from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model
face_cascade = cv2.CascadeClassifier('haar/haarcascade_frontalface_default.xml')
detector2 = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
imgEye = cv2.imread("eye1.png",-1)

# cv2.imshow("Eye Image 4 channel", imgEye)

h,w,c = imgEye.shape[:3]

# print( "**", h, w, c)
orig_mask = imgEye[:,:,3]
# cv2.imshow("orig_mask", orig_mask)

orig_mask_inv = cv2.bitwise_not(orig_mask)
# cv2.imshow("orig_mask_inv", orig_mask_inv)

imgEye = imgEye[:,:,0:3]



class YOLO(object):
    _defaults = {
        "model_path": 'trained_weights_final.h5',
        "anchors_path": 'yolo_anchors.txt',
        "classes_path": 'eye.txt',
        "score" : 0.99,
        "iou" : 0.45,
        "model_image_size" : (640,1280),
        "gpu_num" : 1,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()

        # config = tf.ConfigProto(log_device_placement=True)
        # config.gpu_options.allow_growth = False
        # config.gpu_options.per_process_gpu_memory_fraction = 1
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        #
        # config.gpu_options.allow_growth = True
        # config = tf.ConfigProto(device_count={'GPU': 0})
        # self.sess = tf.Session(config=config)
        # self.sess = tf.Session()
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
        # config = tf.ConfigProto(gpu_options=gpu_options)


        self.sess = tf.Session()
        K.set_session(self.sess)
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'



        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image,frame,x,y):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')


        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })



        font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            # draw = ImageDraw.Draw(image)
            # label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5))
            left = max(0, np.floor(left + 0.5))
            bottom = min(image.size[1], np.floor(bottom + 0.5))
            right = min(image.size[0], np.floor(right + 0.5))


            crop_img1 = frame[int(top+y):int(bottom+y), int(left+x):int(right+x)]
#
#
# ######################################################################################
            eyeOverlayHeight, eyeOverlayWidth, channels = crop_img1.shape
            # print("sdfsadfadf",eyeOverlayHeight,eyeOverlayWidth)

            # h, w, c = imgEye.shape[:3]
            # print("**", h, w, c)

            eyeOverlay = cv2.resize(imgEye, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
            # cv2.imshow("eyeOverlay", crop_img1)
            # print("######", eyeOverlay.shape)

            mask = cv2.resize(orig_mask, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (eyeOverlayWidth, eyeOverlayHeight), interpolation=cv2.INTER_AREA)
            mask_inv = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # cv2.imshow("mask", mask)
            # cv2.imshow("mask_inv", mask_inv)
            roi = frame[int(top+y):int(bottom+y), int(left+x):int(right+x)]
            # cv2.imshow("mask", roi)
            face_part = (roi * (1 / 255.0)) * (mask_inv * (1 / 255.0))
            # cv2.imshow("mask", face_part)
            overlay_part = (eyeOverlay * (1 / 255.0)) * (mask * (1 / 255.0))
            # cv2.imshow("roi", roi)
            # roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
            # cv2.imshow("bg", roi_bg)
            # print("roi_bg shape :::: ", roi_bg.shape)

            # roi_fg = cv2.bitwise_and(eyeOverlay, eyeOverlay, mask=mask)
            dst = cv2.addWeighted(face_part, 255.0, overlay_part, 255.0, 0.0)
            # cv2.imshow("fg", roi_fg)

            # dst = cv2.add(roi_bg, roi_fg)
            # cv2.imshow("dst", dst)
            # cv2.rectangle(images,(left+x,top+y), (right+x,bottom+y),(255, 0, 0),2)
            frame[int(top+y):int(bottom+y), int(left+x):int(right+x)] = dst
            # cv2.imshow("--", frame)




        return frame

    def close_session(self):
        self.sess.close()

def detect_video(yolo, frame):



    frame1=frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(faces)

    if faces != ():
        roi_color = frame[faces[0][1]:faces[0][1] + faces[0][3], faces[0][0]:faces[0][0] + faces[0][2]]
        image = Image.fromarray(roi_color)

        images = yolo.detect_image(image,frame,faces[0][0],faces[0][1])





        img_gray1 = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
        mask1 = np.zeros_like(img_gray1)
        # detector1 = dlib.get_frontal_face_detector()

        faces1 = detector2(img_gray1)



        for face1 in faces1:
            landmarks1 = predictor(img_gray1, face1)
            landmarks_points1 = []
            for n1 in range(36, 48):
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
            fg = cv2.bitwise_or(images, images, mask=mask1)
            mask_inv = cv2.bitwise_not(mask1)
            bk = cv2.bitwise_or(frame1, frame1, mask=mask_inv)
            final = cv2.bitwise_or(fg, bk)
            return final
            # cv2.imshow("out.jpg", final)

    else:
        return frame1




if __name__ == '__main__':
    import cv2
    cap=cv2.VideoCapture(0)

    frnt = YOLO()
    while True:
        ret, img = cap.read()
        k=detect_video(frnt,img)
        cv2.imshow("out.jpg", k)

