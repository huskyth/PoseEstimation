import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
import torch
import sys
import argparse
from mvn.utils import cfg
# from mvn.utils.img import IMAGENET_MEAN, IMAGENET_STD
from mvn.models.triangulation_trt import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from tqdm import tqdm
import time
import cv2
import numpy as np

from collections import defaultdict
from socket import *
import threading
import json

from videopose.preprocess import prep_frame
from mvn.utils.img import resize_image, crop_image, normalize_image
from videopose.dataloader import changeBox
from videopose.img import to_torch
from mvn.datasets import utils as dataset_utils
from videopose.yolo.models.common import DetectMultiBackend
from videopose.yolo.utils.general import (LOGGER, Profile, check_file, check_img_size, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, box_are,track_box,xyxy2xywh,
                           fast_color_histogram,compare_img,track_boxcolor)
from videopose.smoothnet.smoothnet import SmoothNet
from videopose.smoothnet.utils import window_to_seq_only_last
from videopose import h36m_skeleton_re
from send_msg import StartTCP, SendMsg
from get_cap import GetCap
from mvn.utils.read_camera import get_parameters

from mvn.utils.multiview import Camera
from tqdm import tqdm
import time
from pathlib import Path

from savgol_filer import SAVGOLFilter
import copy


torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

num_flame = 30
FPS = 12
msg = None
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='experiments/human36m/eval/human36m_alg.yaml', help="Path, where config file is stored")
    parser.add_argument('--inp_dim', dest='inp_dim', type=str, default='384', help='inpdim')
    parser.add_argument("--det_config", type=str, default='videopose/yolo/yolov3-spp.cfg', help="Path, where config file is stored")
    parser.add_argument("--det_weights", type=str, default='videopose/yolo/yolov5m.pt', help="Path, where weights file is stored")
    parser.add_argument("--vid1", type=str,
                        default="/home/zjlab/dataset/video_cyj/V1_01.avi",
                        help="Path to video from camera 1")
    parser.add_argument("--vid2", type=str,
                        default="/home/zjlab/wl/real-time-pose-estimation/data/V1_02.avi",
                        help="Path to video from camera 2")
    parser.add_argument("--vid3", type=str,
                        default="F:/video-823/V1_03.avi",
                        help="Path to video from camera 3")
    parser.add_argument("--vid4", type=str,
                        default="F:/video-823/V1_04.avi",
                        help="Path to video from camera 4")
    parser.add_argument("--lbls", type=str, default='./human36m-multiview-labels-GTbboxes.npy', help="Path to labels with camera parameters")
    parser.add_argument('--conf_thres', type=float, default=0.2, help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument("--intri", type=str, default="/home/zjlab/calibration/EasyMocap-master/data/extri_data/intri.yml", help="Path to intri parameters")
    parser.add_argument("--extri", type=str, default="/home/zjlab/calibration/EasyMocap-master/data/extri_data/extri.yml", help="Path to extri parameters")


    parser.add_argument('--slide_window_size', type=int, default="8", help='slide window size')
    parser.add_argument('--smooth_weights', type=str, default="videopose/smoothnet/checkpoint.pth.tar",help='pretrained checkpoint file path')



    args = parser.parse_args()
    return args



def det_preproc(streams, inp_dim, msg):
    sample = defaultdict(list)

    for v in range(len(streams)):
        stream = streams[v]
        frame = stream.frame
        # print(frame)
        sample['orig_img'].append(frame)
        # if v == 0:
        #     cv2.imshow('f', frame)
        #     cv2.waitKey(1)

        

    for v in range(len(streams)):
        frame = sample['orig_img'][v]

        # stream = streams[v]
        # grabbed, frame = stream.read()
        # print(frame.shape)
        # # cv2.imwrite('img1.jpg', frame)

        # # if the `grabbed` boolean is `False`, then we have
        # # reached the end of the video file
        # if not grabbed:

        #     print('===========================> can not read video')
        #     sys.stdout.flush()
        #     msg.stop(streams)
        #     # client_string.close()
        #     exit()
        
        # if v == 0:
        #     img_encode = cv2.imencode('.jpg', frame, encode_param)[1]
        #     data = np.array(img_encode)
        #     stringData = data.tostring()
        #     client_string.send(str(len(stringData)).ljust(16))
        #     client_string.send(stringData)

        if v == 0:
            v = v + 1
        else:
            v = v + 2
        retval_camera = Camera(cameras_all['0{}'.format(v)]['R'], cameras_all['0{}'.format(v)]['T'], cameras_all['0{}'.format(v)]['K'],
                            cameras_all['0{}'.format(v)]['dist'], str(v))
        
        # R1 = [[0.235684, 0.970691, -0.047027], [0.380989, -0.136804, -0.914403], [-0.894036, 0.197594, -0.402065]]
        # t1 = [0.280419, 0.576737, 2.790456]
        # K1 = [[1005.963003, 0.000000, 976.153958],
        #       [0.000000, 1003.991297, 576.992033],
        #       [0.000000, 0.000000, 1.000000]]
        # dist1 = [-0.050492, 0.080261, 0.001106, 0.006625, 0.000000]
        #
        # R2 = [[-0.413168, 0.909910, -0.036819], [0.276664, 0.086901, -0.957029], [-0.867611, -0.405600, -0.287644]]
        # t2 = [-0.886750, 0.921617, 2.741842]
        # K2 = [[1006.505518, 0.000000, 1006.348360],
        #       [0.000000, 1003.046864, 526.469265],
        #       [0.000000, 0.000000, 1.000000]]
        # dist2 = [-0.053939, 0.105380, 0.001648, 0.002457, 0.000000]
        #
        # if v == 0:
        #     retval_camera = Camera(R1, t1, K1, dist1, '0')
        # else:
        #     retval_camera = Camera(R2, t2, K2, dist2, '1')

            
        # process and add the frame to the queue
        img_k, orig_img_k, im_dim_list_k = prep_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), inp_dim)

        sample['images'].append(img_k)
        # sample['orig_img'].append(orig_img_k)
        sample['im_dim_list'].append(im_dim_list_k)
        sample['cameras'].append(retval_camera)

    return sample

def vis_bbox(img, bbox, inp_res, orig_res, clr=(255,0,0)):
    x1, y1, x2, y2, conf = bbox[:5]
    orig_h, orig_w = orig_res
    ratio_coeff = orig_h // inp_res
    x1 = x1 * ratio_coeff
    x2 = x2 * ratio_coeff
    y1 = y1 * ratio_coeff - (orig_h - orig_w) // 2
    y2 = y2 * ratio_coeff - (orig_h - orig_w) // 2
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), clr, thickness=2)
    cv2.putText(img, str(conf), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=clr)


def det_process(frame_num, msg=None):
    kpt_s = ['Hip', 'RightHip', 'RightKnee', 'RightAnkle', 'LeftHip', 'LeftKnee', 'LeftAnkle',
             'Spine', 'Thorax', 'Neck', 'Head', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'RightShoulder',
             'RightElbow', 'RightWrist']
    track_result = 2
    # when move camera pred_last need to updata
    # pred_last = [torch.tensor([[189.37500, 119.56250, 235.12500, 253.50000,   0.90430,   0.00000]]),
    #              torch.tensor([[114.00000, 128.12500, 153.50000, 260.00000,   0.90186,   0.00000]])]

    # 8.24 cam
    # pred_last = [torch.tensor([[169.73438, 143.96875, 204.51562, 254.78125,   0.89683,   0.00000]]), torch.tensor([[203.20312, 136.34375, 239.54688, 256.90625,   0.90927,   0.00000]]), 
    #              torch.tensor([[210.32812, 162.50000, 242.42188, 254.25000,   0.79798,   0.00000]]), torch.tensor([[199.03125, 149.65625, 229.46875, 263.84375,   0.87527,   0.00000]])]
    pred_last = [torch.tensor([[192.51562, 102.50000, 251.23438, 268.50000,   0.89683,   0.00000]]), torch.tensor([[ 94.00000, 114.31250, 136.87500, 256.93750,   0.92307,   0.00000]])]
    # pred_last = [torch.tensor([[178.37500, 138.00000, 209.87500, 253.00000, 0.90820, 0.00000]]),
    #  torch.tensor([[184.00000, 134.37500, 217.50000, 253.12500, 0.90674, 0.00000]])]

    track_color = []
    last_img=[]
    stop_track = 0
    pred = []

    last_box_org=[]
    last_point_org=[]
    for i in range(views):
        pred.append(torch.zeros(1, 6))
        last_point_org.append(torch.zeros(2, 17))


    # while True:
    for k_i in tqdm(range(1000)):
        local_times = {}

        # torch.cuda.synchronize()
        local_times['start'] = time.time()

        time.sleep(0.02)
        sample = det_preproc(streams, int(args.inp_dim), msg)

        # continue

        with torch.no_grad():
            # Human Detection
            sample['images'] = torch.cat(sample['images']).to(device)
            imgs = sample['images']
            sample['im_dim_list'] = torch.FloatTensor(sample['im_dim_list']).repeat(1, 2)

            im_dim_list = sample['im_dim_list']

            # torch.cuda.synchronize()
            local_times['before_detection'] = time.time() - local_times['start']
            # print(local_times['before_detection'])
            # continue

            # print(torch.sum(sample['images']))

            yolotime = time.time()
            prediction = det_model(imgs)
            print("yolo:", time.time() - yolotime)


            if track_result==2:
                # last_img = copy.deepcopy(sample['orig_img'])
                # track_result = 1
                # for i in range(len(pred_last)):
                #     a = fast_color_histogram(sample['orig_img'][i], pred_last[i][0])
                #     track_color.append(a)
                
                ### when change camera do it to updata pred_last
                pred_begin = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=100)
                for i in range(len(pred_begin)):
                    for j in range(len(pred_begin[i])):
                        if box_are(pred[i][0]) < box_are(pred_begin[i][j]):
                            pred[i][0] = pred_begin[i][j]
                    print(pred[i][0])
                    a = fast_color_histogram(sample['orig_img'][i], pred_last[i][0])
                    track_color.append(a)
                if box_are(pred[0][0])==0 or box_are(pred[1][0])==0:
                    track_result = 2
                else:
                    print(pred)
                    last_img = sample['orig_img']
                    pred_last = copy.deepcopy(pred)
                    track_result = 1

            if track_result == 0:
                pred_begin = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=100)
                for i, bboxs in enumerate(pred_begin):
                    track_id = compare_img(sample['orig_img'][i], list(bboxs), track_color[i])
                    if track_id == -1:
                        pred = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=1)
                        break
                    else:
                        pred[i][0] = bboxs[track_id]
                pred_last = copy.deepcopy(pred)
                last_img = sample['orig_img']
                track_result = 1

            else:
                aaaa = time.time()
                pred = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=100)
                print("nms track :", time.time() - aaaa)
                # for cam_i in range(len(pred)):    # draw all bboxs
                #     bg = sample['orig_img'][cam_i]
                #     for i, bbox in enumerate(pred[cam_i].tolist()):
                #         vis_bbox(bg, bbox, 384, (1920, 1080), clr=(0, 255, 0))
                
                pred, track_result = track_boxcolor(pred, pred_last, track_color, sample['orig_img'], last_img)
                last_img = sample['orig_img']

                #### visulize video
                for cam_i in range(len(pred)):
                    img = copy.deepcopy(sample['orig_img'][cam_i])
                    # for i, bbox in enumerate(pred[cam_i].tolist()):    # draw track bbox
                    #     vis_bbox(bg, bbox, 384, (1920, 1080), clr=(255, 0, 0))
                    #     vis_bbox(bg, pred_last[cam_i][0], 384, (1920, 1080), clr=(0, 0, 255))
                    for n in range(17):   # draw 2d point
                        cor_x, cor_y = int(last_point_org[cam_i][0, n]), int(last_point_org[cam_i][1, n])
                        # cv2.circle(bg, (cor_x, cor_y), 4, p_color[n], -1)
                        cv2.circle(img, (cor_x, cor_y), 7, (0, 0, 255), -1)
                    
                    for c in connect:    # draw skeleton connect
                        start_x, start_y = int(last_point_org[cam_i][0, c[0]]), int(last_point_org[cam_i][1, c[0]])
                        end_x, end_y = int(last_point_org[cam_i][0, c[1]]), int(last_point_org[cam_i][1, c[1]])
                        cv2.line(img, (start_x, start_y), (end_x, end_y), (0,0,0), thickness=6)

                    img = cv2.resize(img, (2560, 1440))
                    cv2.imshow(str(cam_i), img)
                    cv2.waitKey(1)
                


                if track_result == 0:
                    pred = copy.deepcopy(pred_last)
                    # pred = kalmen_pred.copy()
                    track_result = 1
                    stop_track += 1
                    if stop_track > 4:
                        stop_track = 0
                        track_result = 0
                else:
                    pred_last = copy.deepcopy(pred)
                    stop_track = 0

            local_times['after_detection'] = time.time() - local_times['start']

            boxes = []
            # time1 = time.time()
            # 8ms
            for i, det in enumerate(pred):  # per image
                im0 = sample['orig_img'][i].copy()
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(sample['images'][i].shape[1:], det[:, :4], im0.shape).round()
                    boxes.append(det[:, :4])
            # print('------------', time.time() - time1)

            for k in range(len(sample['orig_img'])):
                boxes_k = boxes[k]
                sample['boxes_k'].append(boxes_k.cpu())
            sample['boxes_k'] = torch.cat(sample['boxes_k'])

            # print(sample['boxes_k'])

            orig_img = sample['orig_img']
            boxes = sample['boxes_k']
            cameras = sample.pop('cameras')

            if boxes is None or boxes.nelement() == 0:
                print('hi there')
                continue

            # torch.cuda.synchronize()
            # local_times['idk2'] = time.time() - local_times['start']
            last_box_org=[]

            for k in range(len(orig_img)):
                inp = orig_img[k]  # cv2.cvtColor(orig_img[k], cv2.COLOR_BGR2RGB)
                image_shape = config.image_shape
                cameras_k = cameras[k]

                boxes_k = boxes[k].unsqueeze(0)
                box = tuple(np.array(boxes_k[0]))
                box = changeBox(box, inp)
                # print("box", k, ":", box)

                last_box_org.append(box) # foe 2d point visual

                inp_s = crop_image(inp, box)  # 7ms

                # inp_s = crop_image_np(inp, box)
                cameras_k.update_after_crop(box)

                image_shape_before_resize = inp_s.shape[:2]

                inp_s = resize_image(inp_s, image_shape)  # 1ms

                sample['inp'].append(inp_s)
                # cv2.imwrite("{}.jpg".format(k), inp_s)
                cameras_k.update_after_resize(image_shape_before_resize, image_shape)

                # 6ms
                inp_s = normalize_image(inp_s)
                inp_s = np.transpose(inp_s, (2, 0, 1))
                inp_s = to_torch(inp_s).float()


                sample['inps'].append(inp_s)
                sample['cameras'].append(cameras_k)


            orig_img = sample['orig_img']
            boxes = sample['boxes_k']
            inp = sample['inp']
            if orig_img is None:
                print(f'{i}-th image read None: handle_video')
                break
            if boxes is None:
                continue

            # for l_i in range(len(inp)):
            #     frames_to_save[l_i].append(orig_img[l_i])
                # fn = os.path.join(temp_path, f"{str(k_i).zfill(4)}_{l_i}.jpg")
                # cv2.imwrite(fn, inp[l_i])

            # Pose Estimation

            images_batch, proj_matricies_batch = dataset_utils.prepare_batch_video(sample, len(videos_paths), device)

            # if i == 1:
            #     torch.save(images_batch, 'images_batch.pt')
            #     torch.save(proj_matricies_batch, 'proj_matricies_batch.pt')
            #     pickle.dump(sample, 'sample.pkl')

            # images_batch = torch.load('images_batch.pt')
            # proj_matricies_batch = torch.load('proj_matricies_batch.pt')
            # sample = torch.load('sample.pt')

            # torch.cuda.synchronize()
            local_times['before_3d'] = time.time() - local_times['start']

            keypoints_3d_pred, keypoints_2d_alg, heatmaps_alg, confidences_alg, lt_t1, lt_t2 = lt_model(images_batch,
                                                                                                        proj_matricies_batch,
                                                                                                        sample)
            
            
            for i, box_org in enumerate(last_box_org):
                xx = (box_org[0]+((keypoints_2d_alg[0,i,:,0])/384*(box_org[2]-box_org[0]))).unsqueeze(0)
                yy = (box_org[1]+((keypoints_2d_alg[0,i,:,1])/384*(box_org[3]-box_org[1]))).unsqueeze(0)
                last_point_org[i]=torch.concat([xx,yy], 0)


            # torch.cuda.synchronize()
            local_times['after_2d'] = lt_t1 - local_times['start']

            # torch.cuda.synchronize()
            local_times['after_3d'] = lt_t2 - local_times['start']

            # key_2ds = keypoints_2d_alg[0]
            # for v in range(len(key_2ds)):
            #     key_2d = key_2ds[v]
            #     bg = cv2.cvtColor(inp[v], cv2.COLOR_RGB2BGR)  # sample['orig_img']
            #     for n in range(17):
            #         cor_x, cor_y = int(key_2d[n][0]), int(key_2d[n][1])
            #         cv2.circle(bg, (cor_x, cor_y), 4, p_color[n], -1)
            #     cv2.imshow('1' + str(v), bg)
            #     cv2.waitKey(1)

            # key_2ds = keypoints_2d_alg[0]
            # for v in range(len(key_2ds)):
            #     key_2d = key_2ds[v]
            #     bg = cv2.cvtColor(inp[v], cv2.COLOR_RGB2BGR)
            #     for n in range(17):
            #         cor_x, cor_y = int(key_2d[n][0]), int(key_2d[n][1])
            #         cv2.circle(bg, (cor_x, cor_y), 4, p_color[n], -1)
            #
            #     frames_to_save[f"{v}_kp"].append(bg)
            #     cv2.imwrite("/data/users/yijia/learnable-triangulation-pytorch-master/img_{}.jpg".format(v), bg)

            if frame_num == 0:
                keypoints_3d = keypoints_3d_pred
                keypoints_3d_pred = keypoints_3d_pred[0]
                predicted_pos = [keypoints_3d_pred]
            else:
                frame_num += 1
                keypoints_3d = torch.cat((keypoints_3d, keypoints_3d_pred), dim=0)
                keypoints_3d_pred = keypoints_3d_pred[0]
                if frame_num > args.slide_window_size:
                    data_pred_window = keypoints_3d[frame_num - 8:frame_num, :]
                    data_pred_window = data_pred_window.permute(0, 2, 1)
                    predicted_pos, time_filter = filter_model(data_pred_window)
                    predicted_pos = predicted_pos.permute(0, 2, 1)

                    pos_last = (predicted_pos[-1, :]+predicted_pos[-2, :])/2
                    predicted_pos = [pos_last, predicted_pos[-1, :]]
                else:
                    predicted_pos = [keypoints_3d_pred]


                    # out_pos2 = fliter_window_to_seq(predicted_pos, args.slide_window_size, mode="out2_mean").reshape(-1, 17, 3)
                    # print("time of filter:", time_filter)

            for predict_pos in predicted_pos:
                point3d = np.array(predict_pos[point_change, :].cpu())
                # point3d = np.array(keypoints_3d_pred.cpu())
                kpts_all = {joint: [] for joint in kpt_s}
                for j in range(len(point3d)):

                    X = point3d[j][0]
                    Y = point3d[j][1]
                    Z = point3d[j][2]

                    point3d[j][0] = -X
                    point3d[j][1] = Z
                    point3d[j][2] = -Y

                if frame_num == 0:
                    bone_len = np.linalg.norm(point3d[2] - point3d[3])
                    frame_num += 1

                unity_len = 0.43
                # unity_len = 0.56
                scale = unity_len / bone_len
                for j in range(len(point3d)):
                    kpts_dict = []
                    # pose_new.append(point3d[j]*scale)
                    kpts_dict.append(float(point3d[j][0]*scale))
                    kpts_dict.append(float(point3d[j][1]*scale))
                    kpts_dict.append(float(point3d[j][2]*scale))
                    kpts_all[kpt_s[j]] = kpts_dict
                if kpts_all is None:
                    msg.save(None)
                msg.save(kpts_all)

            # msg = json.dumps(kpts_all)
            # client.send(msg.encode('utf-8'))

            torch.cuda.synchronize()
            local_times['total'] = time.time() - local_times['start']

        print('*' * 20)
        prev = 0
        for i, k in enumerate(local_times):
            if 'start' in k:
                continue
            print(k, round(local_times[k], 4))
            if i > 1:
                print(round(local_times[k] - prev, 4))
            times[k].append(local_times[k])
            prev = local_times[k]

    print('*' * 40)
    prev = 0
    start_from = 2
    for i, k in enumerate(times):
        if 'start' in k:
            continue
        s = sum(times[k][start_from:])
        l = len(times[k][start_from:])

        if l < 1:
            continue

        print(k, round(s / l, 4))
        if l > 1:
            print(round((s - prev) / l, 4))
        prev = s

def clean_crazy_flame(data_prd):
    data_mean = data_prd.reshape(-1,51).mean(1)
    num_w = 0
    # # 用加速度计算
    for i in range(3,data_prd.shape[0]):
        data_speed = abs(data_mean[i]+data_mean[i-2]-2*data_mean[i-1])
        if data_speed>10:
            num_w = num_w+1
            if num_w>10:
                num_w=0
                continue
            data_prd[i,...] = data_prd[i-1,...]
            data_mean[i] = data_mean[i-1]
            print("we have clean flame:", i)
    return data_prd

def fliter_window_to_seq(slide_window,window_size,mode):  # 这是不使用之前的数据，数据变换哪里也要改
    last = window_size-1
    if mode == "out1":  # out1
        output_len = slide_window.shape[0]
        sequence = [[] for j in range(output_len)]
        for i in range(slide_window.shape[0]):
            sequence[i] = slide_window[i, last, :]
        return torch.stack(sequence)
    elif mode == "out2_mean":  # out2 使用了mean来达到fps翻倍
        output_len = slide_window.shape[0] * 2 - 2
        sequence = [[] for j in range(output_len)]
        j = 0
        for i in range(slide_window.shape[0]-1):
            sequence[j] = slide_window[i, last, :]
            j = j + 1
            sequence[j] = (slide_window[i, last, :]+slide_window[i+1, last, :])/2
            j = j + 1
        return torch.stack(sequence)
    else:
        AttributeError("no such mode")

class read_data():
    def __init__(self):
        self.device = device

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.slide_window_size = args.slide_window_size
        self.slide_window_step = 1

        # detected_truth_data = np.load(os.path.join(test_dataset),allow_pickle=True)
        # self.data = detected_truth_data["arr_0"][num_flame].reshape(-1, 21, 3)  # 不同关节点要换
        # self.data = data

        # self.detected_data = np.concatenate((self.data, np.tile(self.data[0], (self.slide_window_size - 1, 1, 1))),
        #                                     axis=0).reshape(-1, 17, 3)

        # print("input shape:", self.data.shape)
        # print("input shape:", self.data)
        # print(self.detected_data.shape)

        # self.input_dimension = self.detected_data.shape[1] * 3

    def visualize_3d(self, model, data, num_flame):
        keypoint_number = 17
        # data_pred = self.detected_data

        ####去除crazy的数据
        # data_pred = clean_crazy_flame(data_pred)

        # data_len = data_pred.shape[0]
        # data_pred = torch.tensor(data_pred).to(self.device)

        # 这里是把数据切片，做成滑动窗口形式，然后放到model里面.
        # data_pred_window = torch.as_strided(
        #     data_pred, ((data_len - self.slide_window_size) // self.slide_window_step + 1,
        #                 self.slide_window_size, keypoint_number, 3),
        #     (self.slide_window_step * keypoint_number * 3, keypoint_number * 3, 3, 1),
        #     storage_offset=0).reshape(-1, self.slide_window_size, self.input_dimension)
        data_pred_window = data[num_flame-7:num_flame+1, :]
        with torch.no_grad():
            data_pred_window = data_pred_window.permute(0, 2, 1)
            predicted_pos, time = model(data_pred_window)
            # data_pred_window = data_pred_window.permute(0, 2, 1)
            predicted_pos = predicted_pos.permute(0, 2, 1)
            print("predicted_pos:", predicted_pos.shape)
            print("time of filter:", time)

        # 把数据还原为之前的大小
        mode = "out1"  #
        out_pos = fliter_window_to_seq(predicted_pos, self.slide_window_size, mode=mode).reshape(-1, keypoint_number, 3)
        # print("out1 flame:", out_pos.shape)

        mode2 = "out2_mean"  # 输出两倍fps
        out_pos2 = fliter_window_to_seq(predicted_pos, self.slide_window_size, mode=mode2).reshape(-1, keypoint_number,
                                                                                                   3)
        # print("out1 flame:", out_pos2.shape)
        return out_pos



if __name__ == '__main__':
    args = parse_args()

    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127)]
    
    connect = [(0,1),(1,2),(2,6),(5,4),(4,3),(3,6),(6,7),(7,8),(8,16),(9,16),(8,12),(11,12),(10,11),(8,13),(13,14),(14,15)]

    temp_path = './temp_folder/demo_pipeline_2cam/'
    os.makedirs(temp_path, exist_ok=True)
    frames_to_save = defaultdict(list)

    times = defaultdict(list)

    device = torch.device(0)
    config = cfg.load_config(args.config)

    cameras_all = get_parameters(args.intri, args.extri)

    client = StartTCP('10.11.33.117', 9997).start()
    # client_string = StartTCP('10.11.140.9', 9998).start()
    msg = SendMsg(client).start()

    times['global_start'] = [time.time()]

    # Loading 3D human pose estimation model
    lt_model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet
    }[config.model.name](config, device=device).eval().to(device)

    from onnx2tensorRT import TrtModel_yolo
    det_model = TrtModel_yolo(config.tensorRT.engine_detection)

    filter_model = SAVGOLFilter(window_size=args.slide_window_size)
    visualizer = read_data()

    # SmartBody_skeleton = h36m_skeleton_re.H36mSkeleton()
    point_change = [6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10]

    times['models_load'] = [time.time() - times['global_start'][0]]

    videos_paths = [0, 1] # [0, 1, 4, 6]
    views = len(videos_paths)
    # videos_paths = [0, 4, 3, 1]
    # videos_paths = [args.vid1, args.vid2] # , args.vid3, args.vid4]
    streams = []

    # for p in videos_paths:
    #     stream = cv2.VideoCapture(p)
    #     stream.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
    #     stream.set(3, 1920)
    #     stream.set(4, 1080)
    #     streams.append(stream)
    #     assert stream.isOpened(), 'Cannot capture source'
    for p in videos_paths:
        stream = GetCap(p).start()
        streams.append(stream)
        # assert stream.isOpened(), 'Cannot capture source'
    
    # stream_len = int(streams[0].get(cv2.CAP_PROP_FRAME_COUNT))

    # times['models_open_streams'] = [time.time() - times['global_start'][0]]

    camera_labels = np.load(args.lbls, allow_pickle=True).item()

    # times['models_load_lbls'] = [time.time() - times['global_start'][0]]

    frame_num = 0
    # det_process(frame_num)

    det_process(frame_num, msg)

    # np.save(os.path.join(temp_path, 'test_3d_output.npy'), keypoints_3d.cpu(), allow_pickle=True)
    # print(keypoints_3d.shape)

    for l_i in frames_to_save:
        # size = (config.image_shape[0], config.image_shape[1])
        size = (1920, 1080)
        fps = 30
        out_fn = os.path.join(temp_path, f"{str(l_i)}.avi")
        result = cv2.VideoWriter(out_fn,
                                cv2.VideoWriter_fourcc(*'DIVX'),
                                fps, size)
        for i, frame in enumerate(frames_to_save[l_i]):
            result.write(frame)
        result.release()
        
    times['global_end'] = [time.time() - times['global_start'][0]]
    
    # print('*'*40)
    # prev = 0
    # start_from = 2
    # for i, k in enumerate(times):
    #     if 'start' in k:
    #         continue
    #     s = sum(times[k][start_from:])
    #     l = len(times[k][start_from:])
    #
    #     if l < 1:
    #         continue
    #
    #     print(k, round(s / l, 4))
    #     if l > 1:
    #         print(round((s - prev) / l, 4))
    #     prev = s

       
    # et = time.time()

    # print(et - st)
    # print((et - st) / n)

    # det_time = sum(x for x, _, _ in ts)
    # k2d_time = sum(x for _, x, _ in ts)
    # k3d_time = sum(x for _, _, x in ts)

    # print('det', det_time)
    # print('det', det_time / n)
    # print('k2d', k2d_time)
    # print('k2d', k2d_time / n)
    # print('k3d', k3d_time)
    # print('k3d', k3d_time / n)