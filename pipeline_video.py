import os
os.environ["OMP_NUM_THREADS"] = "4"
import torch
import sys
import argparse
from mvn.utils import cfg
# from mvn.utils.img import IMAGENET_MEAN, IMAGENET_STD
from mvn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from tqdm import tqdm
import time
import cv2
import numpy as np

from collections import defaultdict

from videopose.preprocess import prep_frame
from mvn.utils.img import resize_image, crop_image, normalize_image
from videopose.dataloader import changeBox
from videopose.img import to_torch
from mvn.datasets import utils as dataset_utils
from videopose.yolo.models.common import DetectMultiBackend
from videopose.yolo.utils.general import (LOGGER, Profile, check_file, check_img_size, colorstr, cv2,
                                          increment_path, non_max_suppression, print_args, scale_boxes,
                                          strip_optimizer, xyxy2xywh,box_are, fast_color_histogram,
                                          compare_img,track_boxcolor)
from videopose.smoothnet.smoothnet import SmoothNet
from videopose.smoothnet.utils import window_to_seq_only_last
from mvn.utils.read_camera import get_parameters
from savgol_filer import SAVGOLFilter

from mvn.utils.multiview import Camera
from tqdm import tqdm
import time
from pathlib import Path
import copy

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

num_flame = 30
FPS = 12

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='experiments/human36m/eval/human36m_alg.yaml', help="Path, where config file is stored")
    parser.add_argument('--inp_dim', dest='inp_dim', type=str, default='384', help='inpdim')
    parser.add_argument("--det_config", type=str, default='videopose/yolo/yolov3-spp.cfg', help="Path, where config file is stored")
    parser.add_argument("--det_weights", type=str, default='videopose/yolo/yolov5m.pt', help="Path, where weights file is stored")
    parser.add_argument("--vid1", type=str, default="/home/zjlab/real-time-pose-estimation/temp_folder/demo_pipeline_2cam/0.avi", help="Path to video from camera 1")
    parser.add_argument("--vid2", type=str, default="/home/zjlab/real-time-pose-estimation/temp_folder/demo_pipeline_2cam/1.avi", help="Path to video from camera 2")
    parser.add_argument("--vid3", type=str, default="/home/zjlab/dataset/video_xxy/V0_03.avi", help="Path to video from camera 3")
    parser.add_argument("--vid4", type=str, default="/home/zjlab/dataset/video_xxy/V0_04.avi", help="Path to video from camera 4")
    parser.add_argument("--intri", type=str, default="/home/zjlab/calibration/EasyMocap-master/data/extri_data/intri.yml", help="Path to intri parameters")
    parser.add_argument("--extri", type=str, default="/home/zjlab/calibration/EasyMocap-master/data/extri_data/extri.yml", help="Path to extri parameters")
    parser.add_argument("--lbls", type=str, default="C:/Users/ADMIN/Desktop/VideoPose/real-time-pose-estimation/human36m-multiview-labels-GTbboxes.npy", help="Path to labels with camera parameters")
    parser.add_argument('--conf-thres', type=float, default=0.35, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')

    parser.add_argument('--slide_window_size', type=int, default="8", help='slide window size')
    parser.add_argument('--smooth_weights', type=str, default="videopose/smoothnet/checkpoint.pth.tar",help='pretrained checkpoint file path')

    args = parser.parse_args()
    return args


def det_preproc(streams, inp_dim):
    sample = defaultdict(list)

    for v in range(len(streams)):

        stream = streams[v]
        grabbed, frame = stream.read()

        # if the `grabbed` boolean is `False`, then we have
        # reached the end of the video file
        if not grabbed:
            print('===========================> This video get ' + str(k) + ' frames in total.')
            sys.stdout.flush()
            exit()

        retval_camera = Camera(cameras_all['0{}'.format(v+1)]['R'], cameras_all['0{}'.format(v+1)]['T'], cameras_all['0{}'.format(v+1)]['K'],
                            cameras_all['0{}'.format(v+1)]['dist'], str(v))
        
        # if v == 0:
        #     retval_camera = Camera(cameras_all['0{}'.format(v+1)]['R'], cameras_all['0{}'.format(v+1)]['T'], cameras_all['0{}'.format(v+1)]['K'],
        #                     cameras_all['0{}'.format(v+1)]['dist'], str(v))
        # else:
        #     retval_camera = Camera(cameras_all['0{}'.format(4)]['R'], cameras_all['0{}'.format(4)]['T'], cameras_all['0{}'.format(4)]['K'],
        #                     cameras_all['0{}'.format(4)]['dist'], str(v))

            
        # process and add the frame to the queue
        img_k, orig_img_k, im_dim_list_k = prep_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), inp_dim)

        sample['images'].append(img_k)
        sample['orig_img'].append(orig_img_k)
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


if __name__ == '__main__':
    args = parse_args()

    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), 
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127)]

    temp_path = './temp_folder/pipeline_4cam/'
    os.makedirs(temp_path, exist_ok=True)
    frames_to_save = defaultdict(list)

    times = defaultdict(list)

    device = torch.device(0)
    config = cfg.load_config(args.config)

    cameras_all = get_parameters(args.intri, args.extri)

    times['global_start'] = [time.time()]

    # Loading 3D human pose estimation model
    lt_model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet
    }[config.model.name](config, device=device).eval().to(device)

    state_dict = torch.load(config.model.checkpoint)
    for key in list(state_dict.keys()):
        new_key = key.replace("module.", "")
        state_dict[new_key] = state_dict.pop(key)

    lt_model.load_state_dict(state_dict, strict=True)
    print("Successfully loaded pretrained weights for whole model")

    # yolov5
    det_model = DetectMultiBackend(args.det_weights, device=device)
    stride, names, pt = det_model.stride, det_model.names, det_model.pt
    imgsz = check_img_size(int(args.inp_dim), s=stride)

    det_model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup

    filter_model = SAVGOLFilter(window_size=args.slide_window_size)

    times['models_load'] = [time.time() - times['global_start'][0]]

    videos_paths = [args.vid1, args.vid2]#, args.vid3, args.vid4]
    streams = []

    for p in videos_paths:
        stream = cv2.VideoCapture(p)
        streams.append(stream)
        assert stream.isOpened(), 'Cannot capture source'
    
    stream_len = int(streams[0].get(cv2.CAP_PROP_FRAME_COUNT))
    print(stream_len)

    # times['models_open_streams'] = [time.time() - times['global_start'][0]]

    # camera_labels = np.load(args.lbls, allow_pickle=True).item()

    # times['models_load_lbls'] = [time.time() - times['global_start'][0]]
    track_result = 2
    pred_last = [torch.tensor([[192.51562, 102.50000, 251.23438, 268.50000,   0.89683,   0.00000]]), torch.tensor([[ 94.00000, 114.31250, 136.87500, 256.93750,   0.92307,   0.00000]])]

    track_color = []
    last_img=[]
    stop_track = 0
    pred = []

    # last_box_org=[]
    last_point_org=[]

    for i in range(len(videos_paths)):
        pred.append(torch.zeros(1, 6))
        last_point_org.append(torch.zeros(2, 17))

    for k_i in tqdm(range(stream_len)):
        local_times = {}

        torch.cuda.synchronize()
        local_times['start'] = time.time()

        sample = det_preproc(streams, int(args.inp_dim))

        # continue
        with torch.no_grad():
            # Human Detection
            sample['images'] = torch.cat(sample['images']).to(device)
            sample['im_dim_list'] = torch.FloatTensor(sample['im_dim_list']).repeat(1, 2)

            im_dim_list = sample['im_dim_list']

            torch.cuda.synchronize()
            local_times['before_detection'] = time.time() - local_times['start']
            # print(local_times['before_detection'])
            # continue

            # print(torch.sum(sample['images']))
            
            prediction = det_model(sample['images'])
            #torch.cuda.synchronize()
            #local_times['dt1'] = time.time() - local_times['start']

            pred = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0)
            torch.cuda.synchronize()
            local_times['after_detection'] = time.time() - local_times['start']

            if track_result == 2:
                last_img = copy.deepcopy(sample['orig_img'])
                track_result = 1
                ### when change camera do it to updata pred_last
                # pred_begin = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=100)
                # for i in range(len(pred_begin)):
                #     for j in range(len(pred_begin[i])):
                #         if box_are(pred[i][0]) < box_are(pred_begin[i][j]):
                #             pred[i][0] = pred_begin[i][j]
                #     a = fast_color_histogram(sample['orig_img'][i], pred[i][0])
                #     track_color.append(a)
                # if box_are(pred[0][0]) == 0 or box_are(pred[1][0]) == 0:
                #     track_result = 2
                # else:
                #     print(pred)
                #     last_img = copy.deepcopy(sample['orig_img'])
                #     pred_last = copy.deepcopy(pred)
                #     track_result = 1

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
                last_img = copy.deepcopy(sample['orig_img'])
                track_result = 1

            else:
                start_det = time.time()
                pred = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=5)
                print("time of track0:", '%.3f' % (time.time() - start_det))
                # for cam_i in range(len(pred)):
                #     frames_to_save[f"{cam_i}_origimg"].append(
                #         cv2.cvtColor(sample['orig_img'][cam_i], cv2.COLOR_RGB2BGR))
                #     bg = sample['orig_img'][cam_i]
                #     for i, bbox in enumerate(pred[cam_i].tolist()):
                #         vis_bbox(bg, bbox, 384, (1920, 1080), clr=(255, 0, 0))

                pred, track_result = track_boxcolor(pred, pred_last, track_color, sample['orig_img'], last_img)
                last_img = copy.deepcopy(sample['orig_img'])

                # for cam_i in range(len(pred)):
                #     bg = sample['orig_img'][cam_i]
                #     for i, bbox in enumerate(pred[cam_i].tolist()):
                #         vis_bbox(bg, bbox, int(args.inp_dim), (1920, 1080), clr=(255, 0, 0))
                #         vis_bbox(bg, pred_last[cam_i][0], int(args.inp_dim), (1920, 1080), clr=(0, 0, 255))
                #     for n in range(17):  # draw 2d point
                #         cor_x, cor_y = int(last_point_org[cam_i][0, n]), int(last_point_org[cam_i][1, n])
                #         cv2.circle(bg, (cor_x, cor_y), 4, p_color[n], -1)
                #     bg = cv2.resize(bg, (960, 540))
                #     cv2.imshow(str(cam_i), bg)
                #     cv2.waitKey(1)
                    # org_frames_save[f"{cam_i}_origimg"].append(
                    #     cv2.cvtColor(sample['orig_img'][cam_i], cv2.COLOR_RGB2BGR))
                    
                print("time of track1:", '%.3f' % (time.time() - start_det))
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
                print("time of track3:", '%.3f' % (time.time() - start_det))

            boxes = []
            for i, det in enumerate(pred):  # per image
                im0 = sample['orig_img'][i].copy()
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(sample['images'][i].shape[1:], det[:, :4], im0.shape).round()
                    boxes.append(det[:, :4])


            for k in range(len(sample['orig_img'])):
                # print(k)
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
                inp = orig_img[k] # cv2.cvtColor(orig_img[k], cv2.COLOR_BGR2RGB)
                image_shape = config.image_shape
                cameras_k = cameras[k]

                boxes_k = boxes[k].unsqueeze(0)
                box = tuple(np.array(boxes_k[0]))
                box = changeBox(box, inp)

                last_box_org.append(box)

                inp_s = crop_image(inp, box)
                cameras_k.update_after_crop(box)

                image_shape_before_resize = inp_s.shape[:2]
                inp_s = resize_image(inp_s, image_shape)
                sample['inp'].append(inp_s)
                cameras_k.update_after_resize(image_shape_before_resize, image_shape)

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

            for l_i in range(len(inp)):
                frames_to_save[l_i].append(cv2.cvtColor(inp[l_i], cv2.COLOR_RGB2BGR))
                # fn = os.path.join(temp_path, f"{str(k_i).zfill(4)}_{l_i}.jpg")
                # cv2.imwrite(fn, inp[l_i])

            # Pose Estimation
            images_batch, proj_matricies_batch = dataset_utils.prepare_batch_video(sample, len(videos_paths), device)

            keypoints_3d_pred, keypoints_2d_alg, heatmaps_alg, confidences_alg, lt_t1, lt_t2 = lt_model(images_batch,
                                                                                    proj_matricies_batch, sample)
                        #### 2d point to org img
            for i, box_org in enumerate(last_box_org):
                xx = (box_org[0]+((keypoints_2d_alg[0,i,:,0])/384*(box_org[2]-box_org[0]))).unsqueeze(0)
                yy = (box_org[1]+((keypoints_2d_alg[0,i,:,1])/384*(box_org[3]-box_org[1]))).unsqueeze(0)
                last_point_org[i]=torch.concat([xx,yy], 0)


            key_2ds = keypoints_2d_alg[0]
            for v in range(len(key_2ds)):
                key_2d = key_2ds[v]
                bg = cv2.cvtColor(inp[v], cv2.COLOR_RGB2BGR)
                for n in range(17):
                    cor_x, cor_y = int(key_2d[n][0]), int(key_2d[n][1])
                    cv2.circle(bg, (cor_x, cor_y), 4, p_color[n], -1)

                frames_to_save[f"{v}_kp"].append(bg)
                # cv2.imwrite("/data/users/yijia/learnable-triangulation-pytorch-master/img_{}.jpg".format(v), bg)

            if k_i == 0:
                keypoints_3d = keypoints_3d_pred
                keypoints_3d_pred = keypoints_3d_pred[0]
                predicted_pos = [keypoints_3d_pred]
            else:
                keypoints_3d = torch.cat((keypoints_3d, keypoints_3d_pred), dim=0)
                keypoints_3d_pred = keypoints_3d_pred[0]
                if k_i > args.slide_window_size:
                    data_pred_window = keypoints_3d[k_i - 8:k_i, :]
                    data_pred_window = data_pred_window.permute(0, 2, 1)
                    predicted_pos, time_filter = filter_model(data_pred_window)
                    predicted_pos = predicted_pos.permute(0, 2, 1)
                    predicted_pos = [predicted_pos[-1, :]]
                else:
                    predicted_pos = [keypoints_3d_pred]
            
            if k_i == 0:
                keypoints_3ds = keypoints_3d
            else:
                keypoints_3ds = torch.cat((keypoints_3ds, predicted_pos[0].reshape(1,17,3)), dim=0)    
            # keypoints_3d = torch.cat((keypoints_3d, predicted_pos[0].reshape(1,17,3)), dim=0)

            torch.cuda.synchronize()
            local_times['total'] = time.time() - local_times['start']
        
        print('*'*20)
        prev = 0
        for i, k in enumerate(local_times):
            if 'start' in k:
                continue
            print(k, round(local_times[k], 4))
            if i > 1:
                print(round(local_times[k] - prev, 4))
            times[k].append(local_times[k])
            prev = local_times[k]

    np.save(os.path.join(temp_path, 'output.npy'), keypoints_3ds.cpu(), allow_pickle=True)
    # print(keypoints_3d.shape)

    for l_i in frames_to_save:
        size = (config.image_shape[0], config.image_shape[1])
        fps = 30
        out_fn = os.path.join(temp_path, f"{str(l_i)}.avi")
        result = cv2.VideoWriter(out_fn,
                                cv2.VideoWriter_fourcc(*'DIVX'),
                                fps, size)
        for i, frame in enumerate(frames_to_save[l_i]):
            result.write(frame)
        result.release()
        
    times['global_end'] = [time.time() - times['global_start'][0]]
    
    print('*'*40)
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
