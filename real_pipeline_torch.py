import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
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
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from videopose.smoothnet.smoothnet import SmoothNet
from videopose.smoothnet.utils import window_to_seq_only_last
from videopose import h36m_skeleton_re

from mvn.utils.multiview import Camera
from tqdm import tqdm
import time
from pathlib import Path

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

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
    parser.add_argument("--vid1", type=str,
                        default="C:/Users/ADMIN/Desktop/VideoPose/real-time-pose-estimation/outputs/video01.avi",
                        help="Path to video from camera 1")
    parser.add_argument("--vid2", type=str,
                        default="C:/Users/ADMIN/Desktop/VideoPose/real-time-pose-estimation/outputs/video02.avi",
                        help="Path to video from camera 2")
    parser.add_argument("--vid3", type=str, default='/OUTPUT/wl/videos/Directions 1.58860488.mp4', help="Path to video from camera 3")
    parser.add_argument("--vid4", type=str, default='/OUTPUT/wl/videos/Directions 1.60457274.mp4', help="Path to video from camera 4")
    parser.add_argument("--lbls", type=str, default='C:/Users/ADMIN/Desktop/VideoPose/real-time-pose-estimation/human36m-multiview-labels-GTbboxes.npy', help="Path to labels with camera parameters")
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')

    parser.add_argument('--slide_window_size', type=int, default="32", help='slide window size')
    parser.add_argument('--smooth_weights', type=str, default="videopose/smoothnet/checkpoint.pth.tar",help='pretrained checkpoint file path')



    args = parser.parse_args()
    return args



def det_preproc(streams, inp_dim, client):
    sample = defaultdict(list)

    for v in range(len(streams)):
        time1 = time.time()
        stream = streams[v]
        grabbed, frame = stream.read()
        # cv2.imwrite('img1.jpg', frame)
        print(time.time() - time1)

        # if the `grabbed` boolean is `False`, then we have
        # reached the end of the video file
        if not grabbed:
            print('===========================> This video get ' + str(k) + ' frames in total.')
            sys.stdout.flush()
            client.close()
            exit()

        # shot_camera = camera_labels['cameras'][5, v]
        # retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'],
        #                             shot_camera['dist'], '0')
        R1 = [[-0.031126, -0.999407, 0.014739],
              [-0.490450, 0.002422, -0.871466],
              [0.870913, -0.034354, -0.490234]]
        t1 = [0.816604, 0.740444, 1.757986]
        K1 = [[1005.963003, 0.000000, 976.153958],
              [0.000000, 1003.991297, 576.992033],
              [0.000000, 0.000000, 1.000000]]
        dist1 = [-0.050492, 0.080261, 0.001106, 0.006625, 0.000000]

        R2 = [[0.291202, -0.956506, 0.017229],
              [-0.397764, -0.137436, -0.907136],
              [0.870049, 0.257307, -0.420485]]
        t2 = [-1.497282, 1.000794, 2.141635]
        K2 = [[1006.505518, 0.000000, 1006.348360],
              [0.000000, 1003.046864, 526.469265],
              [0.000000, 0.000000, 1.000000]]
        dist2 = [-0.053939, 0.105380, 0.001648, 0.002457, 0.000000]

        if v == 0:
            retval_camera = Camera(R1, t1, K1, dist1, '0')
        else:
            retval_camera = Camera(R2, t2, K2, dist2, '1')

            
        # process and add the frame to the queue
        img_k, orig_img_k, im_dim_list_k = prep_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), inp_dim)

        sample['images'].append(img_k)
        sample['orig_img'].append(orig_img_k)
        sample['im_dim_list'].append(im_dim_list_k)
        sample['cameras'].append(retval_camera)

    return sample

def det_process(client, frame_num):
    kpt_s = ['Hip', 'RightHip', 'RightKnee', 'RightAnkle', 'LeftHip', 'LeftKnee', 'LeftAnkle',
             'Spine', 'Thorax', 'Neck', 'Head', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'RightShoulder',
             'RightElbow', 'RightWrist']
    while True:
        local_times = {}

        torch.cuda.synchronize()
        local_times['start'] = time.time()

        sample = det_preproc(streams, int(args.inp_dim), client)

        # continue

        with torch.no_grad():
            # Human Detection
            sample['images'] = torch.cat(sample['images']).to(device)
            imgs = sample['images']
            sample['im_dim_list'] = torch.FloatTensor(sample['im_dim_list']).repeat(1, 2)

            im_dim_list = sample['im_dim_list']

            torch.cuda.synchronize()
            local_times['before_detection'] = time.time() - local_times['start']
            # print(local_times['before_detection'])
            # continue

            # print(torch.sum(sample['images']))

            prediction = det_model(imgs)

            pred = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0)

            boxes = []
            for i, det in enumerate(pred):  # per image
                im0 = sample['orig_img'][i].copy()
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(sample['images'][i].shape[1:], det[:, :4], im0.shape).round()
                    boxes.append(det[:, :4])

            torch.cuda.synchronize()
            local_times['after_detection'] = time.time() - local_times['start']

            # dets = dynamic_write_results(prediction, 0.05, 80, nms=True, nms_conf=0.6)
            # if isinstance(dets, int) or dets.shape[0] == 0:
            #     print('meh')
            #     continue
            # dets = dets.cpu()
            # im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
            # scaling_factor = torch.min(det_inp_dim / im_dim_list, 1)[0].view(-1, 1)
            #
            # torch.cuda.synchronize()
            # local_times['idk'] = time.time() - local_times['start']
            #
            # # coordinate transfer
            # dets[:, [1, 3]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
            # dets[:, [2, 4]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
            #
            # dets[:, 1:5] /= scaling_factor
            # for j in range(dets.shape[0]):
            #     dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
            #     dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
            # boxes = dets[:, 1:5]
            # scores = dets[:, 5:6]

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

            for k in range(len(orig_img)):
                inp = orig_img[k]  # cv2.cvtColor(orig_img[k], cv2.COLOR_BGR2RGB)
                image_shape = config.image_shape
                cameras_k = cameras[k]

                boxes_k = boxes[k].unsqueeze(0)
                box = tuple(np.array(boxes_k[0]))
                box = changeBox(box, inp)

                inp_s = crop_image(inp, box)
                # inp_s = crop_image_np(inp, box)
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

            # print('*'*30)
            # print(sample['pred_keypoints_3d'])
            # print(sample['cameras'])
            # print(sample['keypoints_3d'])
            # print('*30')

            # if i == 1:
            #     torch.save(images_batch, 'images_batch.pt')
            #     torch.save(proj_matricies_batch, 'proj_matricies_batch.pt')
            #     pickle.dump(sample, 'sample.pkl')

            # images_batch = torch.load('images_batch.pt')
            # proj_matricies_batch = torch.load('proj_matricies_batch.pt')
            # sample = torch.load('sample.pt')

            torch.cuda.synchronize()
            local_times['before_3d'] = time.time() - local_times['start']

            keypoints_3d_pred, keypoints_2d_alg, heatmaps_alg, confidences_alg, lt_t1, lt_t2 = lt_model(images_batch,
                                                                                                        proj_matricies_batch,
                                                                                                        sample)
            # print(keypoints_3d_pred)
            # if k_i > 2:
            #     break

            torch.cuda.synchronize()
            local_times['after_2d'] = lt_t1 - local_times['start']

            torch.cuda.synchronize()
            local_times['after_3d'] = lt_t2 - local_times['start']

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
            point3d = np.array(keypoints_3d_pred[0][point_change, :].cpu())
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
                keypoints_3d = keypoints_3d_pred
                bone_len = np.linalg.norm(point3d[2] - point3d[3])
            else:
                # if k_i % 4 == 0:
                keypoints_3d = torch.cat((keypoints_3d, keypoints_3d_pred), dim=0)

            unity_len = 0.43
            scale = unity_len / bone_len
            for j in range(len(point3d)):
                kpts_dict = []
                # pose_new.append(point3d[j]*scale)
                kpts_dict.append(float(point3d[j][0]*scale))
                kpts_dict.append(float(point3d[j][1]*scale))
                kpts_dict.append(float(point3d[j][2]*scale))
                kpts_all[kpt_s[j]] = kpts_dict
            msg = json.dumps(kpts_all)
            client.send(msg.encode('utf-8'))

            frame_num += 1
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

class read_data():
    def __init__(self, test_dataset,num_flame):
        self.device = device

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.slide_window_size = args.slide_window_size
        self.slide_window_step = 1

        self.data = test_dataset.reshape(-1, 17, 3)
        self.detected_data = np.concatenate((self.data, np.tile(self.data[0], (self.slide_window_size - 1, 1, 1))),
                                            axis=0)

        self.input_dimension = self.detected_data.shape[1]*3

    def visualize_3d(self, model):
        keypoint_number = self.input_dimension//3
        data_pred = self.detected_data

        ####去除crazy的数据
        data_pred = clean_crazy_flame(data_pred)

        data_len = data_pred.shape[0]
        data_pred = torch.tensor(data_pred).to(self.device)

        # 这里是把数据切片，做成滑动窗口形式，然后放到model里面.
        data_pred_window = torch.as_strided(
            data_pred, ((data_len - self.slide_window_size) // self.slide_window_step + 1,
                        self.slide_window_size, keypoint_number, 3),
            (self.slide_window_step * keypoint_number * 3, keypoint_number * 3, 3, 1),
            storage_offset=0).reshape(-1, self.slide_window_size, self.input_dimension)

        with torch.no_grad():
            data_pred_window = data_pred_window.permute(0, 2, 1)
            predicted_pos = model(data_pred_window)
            data_pred_window = data_pred_window.permute(0, 2, 1)
            predicted_pos = predicted_pos.permute(0, 2, 1)

        # 把数据还原为之前的大小
        mode2 = "out2_pred"  # 输出两倍fps
        predicted_pos2 = window_to_seq_only_last(predicted_pos, self.slide_window_size, mode=mode2).reshape(-1, keypoint_number, 3)
        print("out2 flame:",predicted_pos2.shape)

        save_name = mode2+"_3D.npy"

        np.save(os.path.join(temp_path, save_name), predicted_pos2.cpu().numpy())
        print("out2 name:",save_name)

if __name__ == '__main__':
    args = parse_args()

    # p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
    #            (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
    #            (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127)]

    temp_path = '/data/users/wl/real-time-pose-estimation/temp_folder/real_pipeline_2cam/'
    os.makedirs(temp_path, exist_ok=True)
    frames_to_save = defaultdict(list)

    times = defaultdict(list)

    device = torch.device(0)
    config = cfg.load_config(args.config)

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


    # Loading detection model -- yolov3
    # det_model = Darknet(args.det_config).eval().to(device)
    #
    # det_model.load_weights(args.det_weights)
    # det_model.net_info['height'] = args.inp_dim
    # det_inp_dim = int(det_model.net_info['height'])
    # assert det_inp_dim % 32 == 0
    # assert det_inp_dim > 32

    # yolov5
    det_model = DetectMultiBackend(args.det_weights, device=device)
    stride, names, pt = det_model.stride, det_model.names, det_model.pt
    imgsz = check_img_size(int(args.inp_dim), s=stride)

    det_model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # SmoothNet
    smooth_model = SmoothNet(window_size=args.slide_window_size,
                      output_size=2,
                      hidden_size=128,
                      res_hidden_size=32,
                      num_blocks=5,
                      dropout=0.25).to(device)

    # SmartBody_skeleton = h36m_skeleton_re.H36mSkeleton()
    point_change = [6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10]


    times['models_load'] = [time.time() - times['global_start'][0]]

    videos_paths = [0, 1]
    # videos_paths = [args.vid1, args.vid2] # , args.vid3, args.vid4]
    streams = []

    for p in videos_paths:
        stream = cv2.VideoCapture(p)
        stream.set(3, 1920)
        stream.set(4, 1080)
        streams.append(stream)
        assert stream.isOpened(), 'Cannot capture source'
    
    # stream_len = int(streams[0].get(cv2.CAP_PROP_FRAME_COUNT))

    # times['models_open_streams'] = [time.time() - times['global_start'][0]]

    camera_labels = np.load(args.lbls, allow_pickle=True).item()

    # times['models_load_lbls'] = [time.time() - times['global_start'][0]]

    frame_num = 0

    tcp_server = socket(AF_INET, SOCK_STREAM)
    address = ('10.11.140.9', 9999)
    tcp_server.bind(address)
    tcp_server.listen(128)
    print('-------------- Start listening to port 9999 --------------')
    while True:
        client, addr = tcp_server.accept()
        print('-------------- Start listening to port 9999 --------------')
        t = threading.Thread(target=det_process, args=(client, frame_num))
        t.start()



    # visualizer = read_data(keypoints_3d.cpu(), num_flame)
    #
    # if args.smooth_weights != '' and os.path.isfile(
    #         args.smooth_weights):
    #     checkpoint = torch.load(args.smooth_weights)
    #     smooth_model.load_state_dict(checkpoint['state_dict'])
    #     print(f'==> Loaded pretrained model from {args.smooth_weights}...')
    # else:
    #     print(f'{args.smooth_weights} is not a pretrained model!!!!')
    #     exit()
    #
    # visualizer.visualize_3d(smooth_model)
    #
    #
    # np.save(os.path.join(temp_path, 'test_3d_output.npy'), keypoints_3d.cpu(), allow_pickle=True)
    # print(keypoints_3d.shape)

    for l_i in frames_to_save:
        size = (config.image_shape[0], config.image_shape[1])
        fps = 50
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