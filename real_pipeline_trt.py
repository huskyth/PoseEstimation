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
    parser.add_argument("--intri", type=str, default="/home/zjlab/calibration/EasyMocap-master-4/data/extri_data/intri.yml", help="Path to intri parameters")
    parser.add_argument("--extri", type=str, default="/home/zjlab/calibration/EasyMocap-master-4/data/extri_data/extri.yml", help="Path to extri parameters")


    parser.add_argument('--slide_window_size', type=int, default="8", help='slide window size')
    parser.add_argument('--smooth_weights', type=str, default="videopose/smoothnet/checkpoint.pth.tar",help='pretrained checkpoint file path')



    args = parser.parse_args()
    return args


# read camera, one image at one time. And read camera parameters
def det_preproc(streams, inp_dim, msg):
    sample = defaultdict(list)

    for v in range(len(streams)):
        stream = streams[v]
        frame = stream.frame
        sample['orig_img'].append(frame)

    
    for v in range(len(streams)):
        frame = sample['orig_img'][v]

        v = v + 1
        retval_camera = Camera(cameras_all['0{}'.format(v)]['R'], cameras_all['0{}'.format(v)]['T'], cameras_all['0{}'.format(v)]['K'],
                            cameras_all['0{}'.format(v)]['dist'], str(v))

        # process and add the frame to the queue
        img_k, orig_img_k, im_dim_list_k = prep_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), inp_dim)

        sample['images'].append(img_k)
        sample['im_dim_list'].append(im_dim_list_k)
        sample['cameras'].append(retval_camera)

    return sample

# visualize the bounding box from yolo
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

    # if recalibration, you need to updata pred_last by line 190 to line 205 and commenr out line 183 to 187. 
    pred_last = [torch.tensor([[179.82812, 114.62500, 231.92188, 260.87500,   0.86751,   0.00000]]), 
     torch.tensor([[ 58.25000,  90.00000, 134.25000, 258.75000,   0.92904,   0.00000]])]

    track_color = []
    last_img=[]
    stop_track = 0
    pred = []

    last_box_org=[]
    last_point_org=[]
    for i in range(views):
        pred.append(torch.zeros(1, 6))
        last_point_org.append(torch.zeros(2, 17))


    while True:  ## frames_to_save need comment
    # for k_i in tqdm(range(300)):  # if want to save video, use this for save 300 flame. and frames_to_save need dismiss the comment
        local_times = {}

        local_times['start'] = time.time()

        # deal with the input images and cameras
        time.sleep(0.02)
        sample = det_preproc(streams, int(args.inp_dim), msg)

        with torch.no_grad():
            # Human Detection
            sample['images'] = torch.cat(sample['images']).to(device)
            imgs = sample['images']
            sample['im_dim_list'] = torch.FloatTensor(sample['im_dim_list']).repeat(1, 2)

            im_dim_list = sample['im_dim_list']

            local_times['before_detection'] = time.time() - local_times['start']

            # human detection with yolov5
            yolotime = time.time()
            prediction = det_model(imgs)
            print("yolo time", time.time()-yolotime)


            # object tracking
            if track_result==2:
                ### if you updata the pred_last, run this code
                # last_img = copy.deepcopy(sample['orig_img'])
                # track_result = 1
                # for i in range(len(pred_last)):
                #     a = fast_color_histogram(sample['orig_img'][i], pred_last[i][0])
                #     track_color.append(a)
                 
                # ### if recalibration, you need use this code to updata pred_last. and then comment out this code(line 190 to line 205)
                pred_begin = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=100)
                for i in range(len(pred_begin)):
                    for j in range(len(pred_begin[i])):
                        if box_are(pred[i][0]) < box_are(pred_begin[i][j]):
                            pred[i][0] = pred_begin[i][j]
                    print(pred[i][0])
                    a = fast_color_histogram(sample['orig_img'][i], pred[i][0])
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
                nms_time = time.time()
                print("nms time :", nms_time - aaaa)
                pred, track_result = track_boxcolor(pred, pred_last, track_color, sample['orig_img'], last_img)
                last_img = sample['orig_img']

                #### visulize video(line 227 to 248)
                for cam_i in range(len(pred)):
                    img = copy.deepcopy(sample['orig_img'][cam_i])
                    # for i, bbox in enumerate(pred[cam_i].tolist()):    # draw track bbox
                    #     vis_bbox(img, bbox, 384, (1920, 1080), clr=(255, 0, 0))
                    #     vis_bbox(img, pred_last[cam_i][0], 384, (1920, 1080), clr=(0, 0, 255))
                    for n in range(17):   # draw 2d point
                        cor_x, cor_y = int(last_point_org[cam_i][0, n]), int(last_point_org[cam_i][1, n])
                        # cv2.circle(bg, (cor_x, cor_y), 4, p_color[n], -1)
                        cv2.circle(img, (cor_x, cor_y), 7, (0, 0, 255), -1)
                    
                    for c in connect:    # draw skeleton connect
                        start_x, start_y = int(last_point_org[cam_i][0, c[0]]), int(last_point_org[cam_i][1, c[0]])
                        end_x, end_y = int(last_point_org[cam_i][0, c[1]]), int(last_point_org[cam_i][1, c[1]])
                        cv2.line(img, (start_x, start_y), (end_x, end_y), (0,0,0), thickness=6)

                    img = cv2.resize(img, (1920, 1080))
                    # frames_to_save["orig_img"+str(cam_i)].append(sample['orig_img'][cam_i])  # save video
                    # frames_to_save["ske_img"+str(cam_i)].append(img) # save video
                    cv2.imshow(str(cam_i), img)
                    cv2.waitKey(1)
                    
                
                if track_result == 0:
                    pred = copy.deepcopy(pred_last)
                    track_result = 1
                    stop_track += 1
                    if stop_track > 4:
                        stop_track = 0
                        track_result = 0
                else:
                    pred_last = copy.deepcopy(pred)
                    stop_track = 0

            print("time track :", time.time() - nms_time)

            local_times['after_detection'] = time.time() - local_times['start']


            ## resize bounding box
            boxes = []
            for i, det in enumerate(pred):  # per image
                im0 = sample['orig_img'][i].copy()
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(sample['images'][i].shape[1:], det[:, :4], im0.shape).round()
                    boxes.append(det[:, :4])

            for k in range(len(sample['orig_img'])):
                boxes_k = boxes[k]
                sample['boxes_k'].append(boxes_k.cpu())
            sample['boxes_k'] = torch.cat(sample['boxes_k'])

            orig_img = sample['orig_img']
            boxes = sample['boxes_k']
            cameras = sample.pop('cameras')

            if boxes is None or boxes.nelement() == 0:
                print('hi there')
                continue
            
            last_box_org=[]

            # crop the images with bounding box and adjust the camera parameters
            for k in range(len(orig_img)):
                inp = orig_img[k]  # cv2.cvtColor(orig_img[k], cv2.COLOR_BGR2RGB)
                image_shape = config.image_shape
                cameras_k = cameras[k]

                boxes_k = boxes[k].unsqueeze(0)
                box = tuple(np.array(boxes_k[0]))
                box = changeBox(box, inp)

                last_box_org.append(box) # foe 2d point visual

                inp_s = crop_image(inp, box)  # 7ms
                cameras_k.update_after_crop(box)

                image_shape_before_resize = inp_s.shape[:2]

                inp_s = resize_image(inp_s, image_shape)  # 1ms

                sample['inp'].append(inp_s)
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

            # Pose Estimation
            images_batch, proj_matricies_batch = dataset_utils.prepare_batch_video(sample, len(videos_paths), device)

            local_times['before_3d'] = time.time() - local_times['start']

            keypoints_3d_pred, keypoints_2d_alg, heatmaps_alg, confidences_alg, lt_t1, lt_t2 = lt_model(images_batch,
                                                                                                        proj_matricies_batch,
                                                                                                        sample)
            
            ## project 2d point to the original image, only for visualize
            for i, box_org in enumerate(last_box_org):
                xx = (box_org[0]+((keypoints_2d_alg[0,i,:,0])/384*(box_org[2]-box_org[0]))).unsqueeze(0)
                yy = (box_org[1]+((keypoints_2d_alg[0,i,:,1])/384*(box_org[3]-box_org[1]))).unsqueeze(0)
                last_point_org[i]=torch.concat([xx,yy], 0)

            local_times['after_2d'] = lt_t1 - local_times['start']

            local_times['after_3d'] = lt_t2 - local_times['start']

            # send 3d keypoints to unity
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
                    predicted_pos, time_filter = filter_model(data_pred_window)  # double fps only for unity visualize
                    predicted_pos = predicted_pos.permute(0, 2, 1)

                    pos_last = (predicted_pos[-1, :]+predicted_pos[-2, :])/2
                    predicted_pos = [pos_last, predicted_pos[-1, :]]
                else:
                    predicted_pos = [keypoints_3d_pred]

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
                    kpts_dict.append(float(point3d[j][0]*scale))
                    kpts_dict.append(float(point3d[j][1]*scale))
                    kpts_dict.append(float(point3d[j][2]*scale))
                    kpts_all[kpt_s[j]] = kpts_dict
                if kpts_all is None:
                    msg.save(None)
                msg.save(kpts_all)

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
    return keypoints_3d



if __name__ == '__main__':
    args = parse_args()

    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127)]
    
    connect = [(0,1),(1,2),(2,6),(5,4),(4,3),(3,6),(6,7),(7,8),(8,16),(9,16),(8,12),(11,12),(10,11),(8,13),(13,14),(14,15)]

    # temp_path = '/home/zjlab/dataset/video_paper_10.30/video6/'
    temp_path = './temp_folder/demo_pipeline_2cam/'
    os.makedirs(temp_path, exist_ok=True)
    frames_to_save = defaultdict(list)

    times = defaultdict(list)

    device = torch.device(0)
    config = cfg.load_config(args.config)

    cameras_all = get_parameters(args.intri, args.extri)

    client = StartTCP('10.11.140.36', 9997).start()
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
    point_change = [6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10]

    times['models_load'] = [time.time() - times['global_start'][0]]

    videos_paths = [0, 2]
    views = len(videos_paths)
    # videos_paths = [args.vid1, args.vid2] # , args.vid3, args.vid4]
    streams = []

    for p in videos_paths:
        stream = GetCap(p).start()
        streams.append(stream)
        # assert stream.isOpened(), 'Cannot capture source'
    
    # stream_len = int(streams[0].get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = 0

    keypoints_3d = det_process(frame_num, msg)

    np.save(os.path.join(temp_path, "3D-Pose.npy"), keypoints_3d.cpu(), allow_pickle=True)

    for l_i in frames_to_save:
        # size = (config.image_shape[0], config.image_shape[1]) 
        size = (1920, 1080)
        fps = 15
        out_fn = os.path.join(temp_path, f"{str(l_i)}.avi")
        result = cv2.VideoWriter(out_fn,
                                 #cv2.VideoWriter_fourcc('I', '4', '2', '0'),
                                cv2.VideoWriter_fourcc(*'DIVX'),
                                fps, size)
        for i, frame in enumerate(frames_to_save[l_i]):
            result.write(frame)
        result.release()
        
    times['global_end'] = [time.time() - times['global_start'][0]]
    