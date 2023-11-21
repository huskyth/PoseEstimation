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
    def __init__(self, paths, args, batchSize=1, queueSize=50):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.args = args
        self.paths = paths
        stream_list = []
        for path in self.paths:
            stream = cv2.VideoCapture(path)
            stream_list.append(stream)
            assert stream.isOpened(), 'Cannot capture source'
        self.stream = stream_list[0]
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
        self.stopped = False
        stream_list = []
        for path in self.paths:
            stream = cv2.VideoCapture(path)
            stream_list.append(stream)
            assert stream.isOpened(), 'Cannot capture source'

        for i in range(self.num_batches):
            sample = defaultdict(list)
            for k in range(i * self.batchSize, min((i + 1) * self.batchSize, self.datalen)):
                inp_dim = int(self.args.inp_dim)
                for v in range(len(self.paths)):
                    stream = stream_list[v]
                    (grabbed, frame) = stream.read()
                    # if the `grabbed` boolean is `False`, then we have
                    # reached the end of the video file
                    if not grabbed:
                        self.Q.put((None, None, None, None))
                        print('===========================> This video get ' + str(k) + ' frames in total.')
                        sys.stdout.flush()
                        return
                    # process and add the frame to the queue
                    shot_camera = self.labels['cameras'][5, v]
                    retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'],
                                             shot_camera['dist'], '0')
                    # process and add the frame to the queue
                    img_k, orig_img_k, im_dim_list_k = prep_frame(frame, inp_dim)

                    sample['images'].append(img_k)
                    sample['orig_img'].append(orig_img_k)
                    sample['im_dim_list'].append(im_dim_list_k)
                    sample['cameras'].append(retval_camera)

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

            with torch.no_grad():
                # Human Detection
                img = img.cuda()
                prediction = self.det_model(img, CUDA=True)
                # NMS process
                dets = dynamic_write_results(prediction, 0.05, 80, nms=True, nms_conf=0.6)
                if isinstance(dets, int) or dets.shape[0] == 0:
                    for k in range(len(orig_img)):
                        if self.Q.full():
                            time.sleep(2)
                        self.Q.put(None)
                    continue
                dets = dets.cpu()
                im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
                scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

                # coordinate transfer
                dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
                dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

                dets[:, 1:5] /= scaling_factor
                for j in range(dets.shape[0]):
                    dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                    dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]

            for k in range(len(orig_img)):
                boxes_k = boxes[dets[:, 0] == k]
                sample['boxes_k'].append(boxes_k)
            with torch.no_grad():
                sample['boxes_k'] = torch.cat(sample['boxes_k'])
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
                sample = self.detectionLoader.read()
                orig_img = sample['orig_img']
                boxes = sample['boxes_k']
                cameras = sample.pop('cameras')
                if orig_img is None:
                    self.Q.put(None)
                    return
                if boxes is None or boxes.nelement() == 0:
                    while self.Q.full():
                        time.sleep(0.2)
                    self.Q.put(None)
                    continue
                for k in range(len(orig_img)):
                    inp = cv2.cvtColor(orig_img[k], cv2.COLOR_BGR2RGB)
                    image_shape = (384, 384)
                    cameras_k = cameras[k]

                    boxes_k = boxes[k].unsqueeze(0)
                    box = tuple(np.array(boxes_k[0]))
                    box = changeBox(box)

                    inp_s = crop_image(inp, box)
                    cameras_k.update_after_crop(box)

                    image_shape_before_resize = inp_s.shape[:2]
                    inp_s = resize_image(inp_s, image_shape)
                    sample['inp'].append(inp_s)
                    # cv2.imwrite("{}.jpg".format(k), inp_s)
                    cameras_k.update_after_resize(image_shape_before_resize, image_shape)

                    inp_s = normalize_image(inp_s)
                    inp_s = np.transpose(inp_s, (2, 0, 1))
                    inp_s = to_torch(inp_s).float()

                    sample['inps'].append(inp_s)
                    sample['cameras'].append(cameras_k)

                while self.Q.full():
                    time.sleep(0.2)
                self.Q.put(sample)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def len(self):
        # return queue len
        return self.Q.qsize()

def changeBox(box, img):
    ht_img = img.shape[0]
    wd_img = img.shape[1]
    upLeft = [float(box[0]), float(box[1])]
    bottomRight = [float(box[2]), float(box[3])]

    ht = int(bottomRight[0] - upLeft[0])
    width = int(bottomRight[1] - upLeft[1])

    box_len = max(ht, width)
    middle = [(upLeft[0]+bottomRight[0])/2, (upLeft[1]+bottomRight[1])/2]

    upLeft[0] = int(middle[0] - box_len / 2)
    upLeft[1] = int(middle[1] - box_len / 2)
    bottomRight[0] = int(middle[0] + box_len / 2)
    bottomRight[1] = int(middle[1] + box_len / 2)

    if upLeft[0] < 0:
        gap = -upLeft[0]
        upLeft[0] = 0
        bottomRight[0] += gap
    if upLeft[1] < 0:
        gap = -upLeft[1]
        upLeft[1] = 0
        bottomRight[1] += gap
    if bottomRight[0] > wd_img - 1:
        gap = bottomRight[0] - wd_img
        bottomRight[0] = wd_img - 1
        upLeft[0] -= gap
    if bottomRight[1] > ht_img - 1:
        gap = bottomRight[1] - ht_img
        bottomRight[1] = ht_img - 1
        upLeft[1] -= gap

    box = (upLeft[0], upLeft[1], bottomRight[0], bottomRight[1])
    return box
