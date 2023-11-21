import os
import sys
import time
from multiprocessing import Queue as pQueue
from threading import Thread

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from .camera import *
from mvn.utils.multiview import Camera
from mvn.utils.img import get_square_bbox, resize_image, crop_image, normalize_image, scale_bbox
from .img import to_torch

# from SPPE.src.utils.eval import getPrediction, getMultiPeakPrediction
from .img import load_image, cropBox, im_to_torch
# from matching import candidate_reselect as matching
# from opt import opt
# from pPose_nms import pose_nms
from .yolo.darknet import Darknet
from .preprocess import prep_image, prep_frame
from .util import dynamic_write_results
from collections import defaultdict

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue, LifoQueue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue, LifoQueue

# if opt.vis_fast:
#     from fn import vis_frame_fast as vis_frame
# else:
#     from fn import vis_frame


class VideoLoader:
    def __init__(self, path, path_t, args, batchSize=1, queueSize=50):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.args = args
        self.path = path
        self.path_t = path_t
        self.stream = cv2.VideoCapture(path)
        self.stream_t = cv2.VideoCapture(path_t)
        assert self.stream.isOpened() and self.stream_t.isOpened(), 'Cannot capture source'
        self.stopped = False

        labels_path = './data/human36m/extra/human36m-multiview-labels-GTbboxes.npy'
        self.labels = np.load(labels_path, allow_pickle=True).item()

        self.batchSize = batchSize
        self.datalen = int(self.stream.get(cv2.CAP_PROP_FRAME_COUNT))
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def length(self):
        return self.datalen

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        stream = cv2.VideoCapture(self.path)
        stream_t = cv2.VideoCapture(self.path_t)
        assert stream.isOpened() and stream_t.isOpened(), 'Cannot capture source'

        for i in range(self.num_batches):
            sample = defaultdict(list)
            for k in range(i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)):
                inp_dim = int(self.args.inp_dim)
                (grabbed, frame) = stream.read()
                (grabbed_t, frame_t) = stream_t.read()
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed or not grabbed_t:
                    self.Q.put((None, None, None, None))
                    print('===========================> This video get ' + str(k) + ' frames in total.')
                    sys.stdout.flush()
                    return
                # process and add the frame to the queue
                shot_camera_k = self.labels['cameras'][5, 1]
                shot_camera_t = self.labels['cameras'][5, 2]
                retval_camera_k = Camera(shot_camera_k['R'], shot_camera_k['t'], shot_camera_k['K'],
                                         shot_camera_k['dist'], '0')
                retval_camera_t = Camera(shot_camera_t['R'], shot_camera_t['t'], shot_camera_t['K'],
                                         shot_camera_t['dist'], '1')
                image_shape_before_resize = frame.shape[:2]
                image_shape = (inp_dim, inp_dim)
                # process and add the frame to the queue
                img_k, orig_img_k, im_dim_list_k = prep_frame(frame, inp_dim)
                img_t, orig_img_t, im_dim_list_t = prep_frame(frame_t, inp_dim)

                # retval_camera_k.update_after_resize(image_shape_before_resize, image_shape)
                # retval_camera_t.update_after_resize(image_shape_before_resize, image_shape)

                sample['images'].append(img_k)
                sample['images'].append(img_t)
                sample['orig_img'].append(orig_img_k)
                sample['orig_img'].append(orig_img_t)
                sample['im_dim_list'].append(im_dim_list_k)
                sample['im_dim_list'].append(im_dim_list_t)
                sample['cameras'].append(retval_camera_k)
                sample['cameras'].append(retval_camera_t)

                # sample['proj_matrices'].append(retval_camera.projection)

            with torch.no_grad():
                # Human Detection
                sample['images'] = torch.cat(sample['images'])
                sample['im_dim_list'] = torch.FloatTensor(sample['im_dim_list']).repeat(1, 2)

            while self.Q.full():
                time.sleep(2)

            self.Q.put(sample)

    def videoinfo(self):
        # indicate the video info
        fourcc = int(self.stream.get(cv2.CAP_PROP_FOURCC))
        fps = self.stream.get(cv2.CAP_PROP_FPS)
        frameSize = (int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return (fourcc, fps, frameSize)

    def getitem(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        return self.Q.qsize()


class DetectionLoader:
    def __init__(self, dataloder, args, batchSize=1, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.args = args
        self.det_model = Darknet("videopose/yolo/yolov3-spp.cfg")
        self.det_model.load_weights('videopose/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = self.args.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()

        labels_path = './data/human36m/extra/human36m-multiview-labels-GTbboxes.npy'
        self.labels = np.load(labels_path, allow_pickle=True).item()

        self.stopped = False
        self.dataloder = dataloder
        self.batchSize = batchSize
        self.datalen = self.dataloder.length()
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream

        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()

        return self

    def update(self):
        # keep looping the whole dataset
        for i in range(self.num_batches):
            sample = self.dataloder.getitem()
            # img, orig_img, im_name, im_dim_list = self.dataloder.getitem()
            img = sample['images']
            orig_img = sample['orig_img']
            im_dim_list = sample['im_dim_list']
            if img is None:
                self.Q.put((None, None, None, None, None, None, None))
                return

            shot = np.array([[215, 294, 621, 700],[181, 445, 569, 833]])
            # shot = np.array([[245, 128, 775, 658], [241, 277, 643, 629]])


            for k in range(len(orig_img)):
                bbox = shot[k][[1, 0, 3, 2]]  # TLBR to LTRB
                bbox_height = bbox[2] - bbox[0]
                if bbox_height == 0:
                    # convention: if the bbox is empty, then this view is missing
                    continue

                # scale the bounding box
                bbox = scale_bbox(bbox, 1.0)
                boxes_k = bbox

                # inps = torch.zeros(boxes_k.size(0), 3, 320, 256)
                # pt1 = torch.zeros(boxes_k.size(0), 2)
                # pt2 = torch.zeros(boxes_k.size(0), 2)
                sample['boxes_k'].append(boxes_k)
                # sample['inps'].append(inps)
                # sample['pt1'].append(pt1)
                # sample['pt2'].append(pt2)
            # with torch.no_grad():
            #     sample['boxes_k'] = torch.cat(sample['boxes_k'])
                # sample['inps'] = torch.cat(sample['inps'])
                # sample['pt1'] = torch.cat(sample['pt1'])
                # sample['pt2'] = torch.cat(sample['pt2'])
            if self.Q.full():
                time.sleep(2)
            self.Q.put(sample)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


class DetectionProcessor:
    def __init__(self, detectionLoader, args, queueSize=1024):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.args = args
        self.detectionLoader = detectionLoader
        self.stopped = False
        self.datalen = self.detectionLoader.datalen

        # initialize the queue used to store data
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        # t = Thread(target=self.update, args=(), daemon=True)
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping the whole dataset
        for i in range(self.datalen):

            with torch.no_grad():
                # (orig_img, boxes, scores, inps, pt1, pt2) = self.detectionLoader.read()
                sample = self.detectionLoader.read()
                orig_img = sample['orig_img']
                boxes = sample['boxes_k']
                # inps = sample.pop('inps')
                # pt1 = sample.pop('pt1')
                # pt2 = sample.pop('pt2')
                cameras = sample.pop('cameras')
                if orig_img is None:
                    self.Q.put(None)
                    return
                if boxes is None :
                    while self.Q.full():
                        time.sleep(0.2)
                    self.Q.put(None)
                    continue
                for k in range(len(orig_img)):
                    # inp_dim = int(self.args.inp_dim)
                    # inp = im_to_torch(cv2.cvtColor(orig_img[k], cv2.COLOR_BGR2RGB))
                    inp = cv2.cvtColor(orig_img[k], cv2.COLOR_BGR2RGB)


                    # image_shape = (320, 256)
                    image_shape = (384, 384)
                    cameras_k = cameras[k]

                    boxes_k = boxes[k]
                    # inps_k = inps[k].unsqueeze(0)
                    # pt1_k = pt1[k].unsqueeze(0)
                    # pt2_k = pt2[k].unsqueeze(0)
                    # inp_s, pt_1, pt_2 = crop_from_dets(inp, boxes_k, inps_k, pt1_k, pt2_k)
                    box = tuple(np.array(boxes_k))
                    inp_s = crop_image(inp, box)
                    cameras_k.update_after_crop(boxes_k)

                    image_shape_before_resize = inp_s.shape[:2]
                    inp_s = resize_image(inp_s, image_shape)
                    if k == 0:
                        cv2.imwrite("inp.jpg", inp_s)
                    else:
                        cv2.imwrite("inp1.jpg", inp_s)
                    sample['inp'].append(inp_s)
                    cameras_k.update_after_resize(image_shape_before_resize, image_shape)

                    inp_s = normalize_image(inp_s)
                    inp_s = np.transpose(inp_s, (2, 0, 1))
                    inp_s = to_torch(inp_s).float()

                    sample['inps'].append(inp_s)
                    # sample['pt1'].append(pt_1)
                    # sample['pt2'].append(pt_2)
                    sample['cameras'].append(cameras_k)

                # with torch.no_grad():
                #     sample['inps'] = torch.cat(sample['inps'])
                #     sample['pt1'] = torch.cat(sample['pt1'])
                #     sample['pt2'] = torch.cat(sample['pt2'])
                while self.Q.full():
                    time.sleep(0.2)
                self.Q.put(sample)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()


def crop_from_dets(img, boxes, inps, pt1, pt2):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)
    for i, box in enumerate(boxes):
        upLeft = torch.Tensor(
            (float(box[0]), float(box[1])))
        bottomRight = torch.Tensor(
            (float(box[2]), float(box[3])))

        ht = bottomRight[1] - upLeft[1]
        width = bottomRight[0] - upLeft[0]

        scaleRate = 0.3

        upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
        upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
        bottomRight[0] = max(
            min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
        bottomRight[1] = max(
            min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

        try:
            inp = cropBox(tmp_img.clone(), upLeft, bottomRight, 320, 256)
            inps[i] = inp
        except IndexError:
            print(tmp_img.shape)
            print(upLeft)
            print(bottomRight)
            print('===')
        pt1[i] = upLeft
        pt2[i] = bottomRight

    return inps, pt1, pt2
