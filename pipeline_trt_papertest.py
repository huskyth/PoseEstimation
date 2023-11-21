import os
# os.environ["OMP_NUM_THREADS"] = "4"
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

from mvn.utils.read_camera import get_parameters

from collections import defaultdict

from videopose.preprocess import prep_frame
from mvn.utils.img import resize_image, crop_image, normalize_image
from videopose.dataloader import changeBox
from videopose.img import to_torch
from mvn.datasets import utils as dataset_utils
from videopose.yolo.models.common import DetectMultiBackend
from videopose.yolo.utils.general import (LOGGER, Profile, check_file, check_img_size, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh,
                                          track_box, fast_color_histogram, box_are, track_boxcolor, compare_img, Kalman_Filtering_2D)
# from videopose.smoothnet.smoothnet import SmoothNet
from videopose.smoothnet.utils import window_to_seq_only_last

from mvn.utils.multiview import Camera
from tqdm import tqdm
import time
from pathlib import Path
from onnx2tensorRT import TrtModel, TrtModel_yolo
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
    parser.add_argument("--vid1", type=str, default="/home/zjlab/dataset/hospital10-13/10_13_person3/hospital_2023_10_13-10_38_30_camera1.avi", help="Path to video from camera 1")
    parser.add_argument("--vid2", type=str, default="/home/zjlab/dataset/hospital10-13/10_13_person3/hospital_2023_10_13-10_38_30_camera2.avi", help="Path to video from camera 2")
    # parser.add_argument("--vid3", type=str, default='/home/zjlab/dataset/video_xxy/V3_03.avi', help="Path to video from camera 3")
    # parser.add_argument("--vid4", type=str, default='/home/zjlab/dataset/video_xxy/V3_04.avi', help="Path to video from camera 4")
    parser.add_argument("--lbls", type=str, default="./human36m-multiview-labels-GTbboxes.npy", help="Path to labels with camera parameters")
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')

    parser.add_argument("--intri", type=str, default="/home/zjlab/calibration/EasyMocap-master/data/extri_data/intri.yml", help="Path to intri parameters")
    parser.add_argument("--extri", type=str, default="/home/zjlab/calibration/EasyMocap-master/data/extri_data/extri.yml", help="Path to extri parameters")

    parser.add_argument('--slide_window_size', type=int, default="32", help='slide window size')
    parser.add_argument('--smooth_weights', type=str, default="videopose/smoothnet/checkpoint.pth.tar",help='pretrained checkpoint file path')



    args = parser.parse_args()
    return args
save_name_3Dpose = 'person3_video7.npy'

from torch.utils.data import Dataset
class Human36MMultiViewDataset(Dataset):
    def __init__(self,
                 h36m_root='/Vol1/dbstore/datasets/Human3.6M/processed/',
                 labels_path='/Vol1/dbstore/datasets/Human3.6M/extra/human36m-multiview-labels-SSDbboxes.npy',
                 pred_results_path=None,
                 image_shape=(256, 256),
                 train=False,
                 test=False,
                 retain_every_n_frames_in_test=1,
                 with_damaged_actions=False,
                 cuboid_side=2000.0,
                 scale_bbox=1.5,
                 norm_image=True,
                 kind="mpii",
                 undistort_images=False,
                 ignore_cameras=[],
                 crop=True
                 ):

        assert train or test, '`Human36MMultiViewDataset` must be constructed with at least ' \
                              'one of `test=True` / `train=True`'
        assert kind in ("mpii", "human36m")

        self.h36m_root = h36m_root
        self.labels_path = labels_path
        self.image_shape = None if image_shape is None else tuple(image_shape)
        self.scale_bbox = scale_bbox
        self.norm_image = norm_image
        self.cuboid_side = cuboid_side
        self.kind = kind
        self.undistort_images = undistort_images
        self.ignore_cameras = ignore_cameras
        self.crop = crop

        self.labels = np.load(labels_path, allow_pickle=True).item()

        n_cameras = len(self.labels['camera_names'])
        assert all(camera_idx in range(n_cameras) for camera_idx in self.ignore_cameras)

        train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        test_subjects = ['S9', 'S11']

        train_subjects = list(self.labels['subject_names'].index(x) for x in train_subjects)
        test_subjects  = list(self.labels['subject_names'].index(x) for x in test_subjects)

        indices = []
        if train:
            mask = np.isin(self.labels['table']['subject_idx'], train_subjects, assume_unique=True)
            indices.append(np.nonzero(mask)[0])
        if test:
            mask = np.isin(self.labels['table']['subject_idx'], test_subjects, assume_unique=True)

            if not with_damaged_actions:
                mask_S9 = self.labels['table']['subject_idx'] == self.labels['subject_names'].index('S9')

                damaged_actions = 'Greeting-2', 'SittingDown-2', 'Waiting-1'
                damaged_actions = [self.labels['action_names'].index(x) for x in damaged_actions]
                mask_damaged_actions = np.isin(self.labels['table']['action_idx'], damaged_actions)

                mask &= ~(mask_S9 & mask_damaged_actions)

            indices.append(np.nonzero(mask)[0][::retain_every_n_frames_in_test])

        self.labels['table'] = self.labels['table'][np.concatenate(indices)]

        self.num_keypoints = 16 if kind == "mpii" else 17
        assert self.labels['table']['keypoints'].shape[1] == 17, "Use a newer 'labels' file"

        self.keypoints_3d_pred = None
        if pred_results_path is not None:
            pred_results = np.load(pred_results_path, allow_pickle=True)
            keypoints_3d_pred = pred_results['keypoints_3d'][np.argsort(pred_results['indexes'])]
            self.keypoints_3d_pred = keypoints_3d_pred[::retain_every_n_frames_in_test]
            assert len(self.keypoints_3d_pred) == len(self), \
                f"[train={train}, test={test}] {labels_path} has {len(self)} samples, but '{pred_results_path}' " + \
                f"has {len(self.keypoints_3d_pred)}. Did you follow all preprocessing instructions carefully?"

    def __len__(self):
        return len(self.labels['table'])

    def __getitem__(self, idx):
        sample = defaultdict(list) # return value
        shot = self.labels['table'][idx]

        subject = self.labels['subject_names'][shot['subject_idx']]
        action = self.labels['action_names'][shot['action_idx']]
        frame_idx = shot['frame_idx']
        action_name = subject+'_'+action

        for camera_idx, camera_name in enumerate(self.labels['camera_names']):
            if camera_idx in self.ignore_cameras:
                continue
            s = shot['bbox_by_camera_tlbr']

            # load image
            image_path = os.path.join(
                self.h36m_root, subject, action, 'imageSequence' + '-undistorted' * self.undistort_images,
                camera_name, 'img_%06d.jpg' % (frame_idx+1))
            assert os.path.isfile(image_path), '%s doesn\'t exist' % image_path
            image = cv2.imread(image_path)
            # print(image.shape)  # 这个image是(1002, 1000, 3)和(1000, 1000, 3)，很奇怪，得resize一下。pipeline已经自带了resize

            # load camera
            shot_camera = self.labels['cameras'][shot['subject_idx'], camera_idx]
            retval_camera = Camera(shot_camera['R'], shot_camera['t'], shot_camera['K'], shot_camera['dist'], camera_name)

            sample['images'].append(image)
            # sample['detections'].append(bbox + (1.0,)) # TODO add real confidences
            # sample['detections'].append(img)  # TODO add real confidences
            sample['cameras'].append(retval_camera)
            sample['proj_matrices'].append(retval_camera.projection)

        # 3D keypoints
        # add dummy confidences
        sample['keypoints_3d'] = np.pad(
            shot['keypoints'][:self.num_keypoints],
            ((0,0), (0,1)), 'constant', constant_values=1.0)

        # save sample's index
        sample['indexes'] = idx
        sample['actname'] = action_name

        if self.keypoints_3d_pred is not None:
            sample['pred_keypoints_3d'] = self.keypoints_3d_pred[idx]
        sample.default_factory = None
        return sample
    

def det_preproc(streams, ids, inp_dim, cameras_all=None):    
    sample = defaultdict(list)
    read_data = val_dataset[ids]
    sample['actname'] = read_data['actname']

    for v in range(len(read_data['cameras'])):
        frame = read_data['images'][v]

        # stream = streams[v]
        # grabbed, frame = stream.read()
        # # if the `grabbed` boolean is `False`, then we have
        # # reached the end of the video file
        # if not grabbed:
        #     print('===========================> This video get ' + str(k) + ' frames in total.')
        #     sys.stdout.flush()
        #     exit()

        # v = v+1

        # retval_camera = Camera(cameras_all['0{}'.format(v+1)]['R'], cameras_all['0{}'.format(v+1)]['T'], cameras_all['0{}'.format(v+1)]['K'],
        #                     cameras_all['0{}'.format(v+1)]['dist'], str(v+1))
        
        # multiview_data = np.load("/home/zjlab/Pavel/repos/real-time-pose-estimation/data/human36m/extra/human36m-multiview-labels-GTbboxes.npy", allow_pickle=True).tolist()
        # subject_name, camera_name, action_name, camera_configs, labels = multiview_data['subject_names'], multiview_data[
        #     'camera_names'], multiview_data['action_names'], multiview_data['cameras'], multiview_data['table']

        # specific_camera_config = camera_configs[subject_name.index("S9")]
        # retval_camera = Camera(specific_camera_config["R"][v], specific_camera_config["t"][v], specific_camera_config["K"][v], specific_camera_config["dist"][v], str(v))
  
        # process and add the frame to the queue
        img_k, orig_img_k, im_dim_list_k = prep_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), inp_dim)
        
        sample['images'].append(img_k)
        sample['orig_img'].append(orig_img_k)
        sample['im_dim_list'].append(im_dim_list_k)
        sample['cameras']=read_data['cameras']
    return sample

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

def vis_bbox(img, bbox, inp_res, orig_res, clr=(255,0,0)):
    if len(bbox)==6:
        x1, y1, x2, y2, conf = bbox[:5]
    else:
        x1, y1, x2, y2 = bbox[:4]
        conf = 0
    orig_h, orig_w = orig_res
    ratio_coeff = orig_h / inp_res
    x1 = x1 * ratio_coeff
    x2 = x2 * ratio_coeff
    y1 = y1 * ratio_coeff - (orig_h - orig_w) // 2
    y2 = y2 * ratio_coeff - (orig_h - orig_w) // 2
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), clr, thickness=2)
    cv2.putText(img, str('%.2f'% conf), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=clr)


if __name__ == '__main__':
    args = parse_args()

    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), 
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127)]
    connect = [(0,1),(1,2),(2,6),(5,4),(4,3),(3,6),(6,7),(7,8),(8,16),(9,16),(8,12),(11,12),(10,11),(8,13),(13,14),(14,15)]

    temp_path = './temp_folder/inference_pipeline_xxy/'
    os.makedirs(temp_path, exist_ok=True)
    frames_to_save = defaultdict(list)
    org_frames_save = defaultdict(list)

    times = defaultdict(list)

    device = torch.device(0)
    config = cfg.load_config(args.config)

    cameras_all = get_parameters(args.intri, args.extri)
    
    ignore_c=[]
    val_dataset = Human36MMultiViewDataset(
        h36m_root=config.dataset.val.h36m_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
        labels_path=config.dataset.val.labels_path,
        with_damaged_actions=config.dataset.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.dataset.val.undistort_images,
        ignore_cameras=ignore_c,
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
    )
    views = 4 - len(ignore_c)

    times['global_start'] = [time.time()]

    # Loading 3D human pose estimation model
    lt_model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet
    }[config.model.name](config, device=device).eval().to(device)

    # yolov5

    det_model = TrtModel_yolo(config.tensorRT.engine_detection)
    # det_model = DetectMultiBackend(args.det_weights, device=device)
    # stride, names, pt = det_model.stride, det_model.names, det_model.pt
    imgsz = args.inp_dim

    # det_model.warmup(imgsz=(1, 3, imgsz, imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    # SmoothNet
    # smooth_model = SmoothNet(window_size=args.slide_window_size,
    #                   output_size=2,
    #                   hidden_size=128,
    #                   res_hidden_size=32,
    #                   num_blocks=5,
    #                   dropout=0.25).to(device)


    times['models_load'] = [time.time() - times['global_start'][0]]

    # videos_paths = [args.vid1, args.vid2] # [args.vid1, args.vid2] #, args.vid3, args.vid4]
    # streams = []


    # for p in videos_paths:
    #     stream = cv2.VideoCapture(p)
    #     streams.append(stream)
    #     assert stream.isOpened(), 'Cannot capture source'
    
    # steam_len = int(streams[0].get(cv2.CAP_PROP_FRAME_COUNT))

    times['models_open_streams'] = [time.time() - times['global_start'][0]]

    camera_labels = 0 # np.load(args.lbls, allow_pickle=True).item()

    # times['models_load_lbls'] = [time.time() - times['global_start'][0]]
    track_result = 2
    pred_last = [torch.tensor([[210.62500, 135.00000, 256.50000, 274.00000,   0.90039,   0.00000]]),
                 torch.tensor([[216.62500, 138.50000, 255.62500, 270.50000,   0.82568,   0.00000]])]

    track_color = []
    last_img=[]
    stop_track = 0
    pred = []

    image_action_name_last = []
    save_3d_pred_point = defaultdict(list)

    # last_box_org=[]
    last_point_org=[]
    for i in range(views):
        pred.append(torch.zeros(1, 6))
        last_point_org.append(torch.zeros(2, 17))

    ####### kalman filter init
    # Kalman_Filtering_2D = Kalman_Filtering_2D(views*2)
    # Kalman_Filtering_2D.initialize()
    # kalmen_input = np.zeros((views*2, 2))
    # kalmen_pred = [torch.zeros(1, 4), torch.zeros(1, 4)]

    
    for k_i in tqdm(range(val_dataset.__len__())):
    # for k_i in tqdm(range(2)):
        local_times = {}

        torch.cuda.synchronize()
        local_times['start'] = time.time()

        sample = det_preproc(val_dataset, k_i, int(args.inp_dim), cameras_all=cameras_all)

        # 保存3d关节点,当换img的时候，重新跟踪。
        if k_i == 0 or str(sample['actname'])!=str(image_action_name_last):
            track_result = 2    



        # continue
        with torch.no_grad():
            # Human Detection
            sample['images'] = torch.cat(sample['images']).to(device)
            # sample['images'] = sample['images'][0]
            # print(sample['images'].shape)
            sample['im_dim_list'] = torch.FloatTensor(sample['im_dim_list']).repeat(1, 2)

            im_dim_list = sample['im_dim_list']

            torch.cuda.synchronize()
            local_times['before_detection'] = time.time() - local_times['start']

            yolotime=time.time()
            print(sample['images'].shape)

            prediction = det_model(sample['images'])
            print(len(prediction), prediction[0].shape)
            torch.cuda.synchronize()
            print("yolo time", time.time()-yolotime)
            #torch.cuda.synchronize()
            #local_times['dt1'] = time.time() - local_times['start']

            # prediction = prediction[0].reshape(views, -1, 85)
            
            # 不使用跟踪，因为测试集跳的太快了，跟踪效果很差，加上smoothnet也很差，因为测试集被抽帧了，没有时间信息，不连贯
            pred = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=1)
            for i, box in enumerate(pred):
                if box.size(0)==0:
                    pred[i] = torch.tensor([[20.3, 20.3, 300.3, 300.3, 0.8, 0.0]])
            for cam_i in range(len(pred)):
                # bg = sample['orig_img'][cam_i]
                vis_img = copy.deepcopy(sample['orig_img'][cam_i])
                # for i, bbox in enumerate(pred[cam_i].tolist()):
                #     vis_bbox(vis_img, bbox, int(args.inp_dim), (1920, 1080), clr=(255, 0, 0))
                #     vis_bbox(vis_img, pred_last[cam_i][0], int(args.inp_dim), (1920, 1080), clr=(0, 0, 255))

                for n in range(17):   # draw 2d point
                    cor_x, cor_y = int(last_point_org[cam_i][0, n]), int(last_point_org[cam_i][1, n])
                    # cv2.circle(bg, (cor_x, cor_y), 4, p_color[n], -1)
                    cv2.circle(vis_img, (cor_x, cor_y), 7, (0, 0, 255), -1)
                
                for c in connect:    # draw skeleton connect
                    start_x, start_y = int(last_point_org[cam_i][0, c[0]]), int(last_point_org[cam_i][1, c[0]])
                    end_x, end_y = int(last_point_org[cam_i][0, c[1]]), int(last_point_org[cam_i][1, c[1]])
                    cv2.line(vis_img, (start_x, start_y), (end_x, end_y), (0,0,0), thickness=6)

                vis_img = cv2.resize(vis_img, (960, 540))
                cv2.imshow(str(cam_i), vis_img)
                cv2.waitKey(1)
                org_frames_save[f"{cam_i}_origimg"].append(
                    cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))


            #prediction = torch.from_numpy(prediction[0].reshape(views, -1, 85)).to(device)
            # pred = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0)
            # aaa = time.time()
            # if track_result==2:  # when change camera do it to updata pred_last
            #     # last_img = copy.deepcopy(sample['orig_img'])
            #     # track_result = 1
            #     pred_begin = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=100)
            #     track_color.clear

            #     print(len(pred_begin))
            #     for i in range(len(pred_begin)):
            #         pred[i][0] = torch.zeros(1, 6)
            #         for j in range(len(pred_begin[i])):
            #             if box_are(pred[i][0]) < box_are(pred_begin[i][j]) and pred_begin[i].size(0)!=0:
            #                 pred[i][0] = pred_begin[i][j]
            #         a = fast_color_histogram(sample['orig_img'][i], pred[i][0])
            #         track_color.append(a)
            #     if box_are(pred[0][0])==0 or box_are(pred[1][0])==0:
            #         track_result = 2
            #     else:
            #         # print(pred)
            #         last_img = copy.deepcopy(sample['orig_img'])
            #         pred_last = copy.deepcopy(pred)
            #         track_result = 1

            # if track_result == 0:
            #     pred_begin = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=100)
            #     for i, bboxs in enumerate(pred_begin):
            #         track_id = compare_img(sample['orig_img'][i], list(bboxs), track_color[i])
            #         if track_id==-1:
            #             pred = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=1)
            #             break
            #         else:
            #             pred[i][0] = bboxs[track_id]
            #     pred_last = copy.deepcopy(pred)
            #     track_result = 1


            # else:
            #     ### kalmen filter maybe use for lose track
            #     # aaaa = time.time()
            #     # for i in range(views):
            #     #     kalmen_input[2 * i][0:2] = pred_last[i][0][0:2]
            #     #     kalmen_input[2 * i + 1][0:2] = pred_last[i][0][2:4]
            #     # kalmen_out = torch.Tensor(Kalman_Filtering_2D.predict(kalmen_input))
            #     # kalmen_pred[0][0] = kalmen_out[0:4]
            #     # kalmen_pred[1][0] = kalmen_out[4:8]
            #     # print("kalmen :", time.time() - aaaa)

            #     aaaa = time.time()
            #     pred = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=100)
            #     print("nms track :", time.time() - aaaa)
            #     # for cam_i in range(len(pred)):
            #     #     bg = sample['orig_img'][cam_i]
            #     #     for i, bbox in enumerate(pred[cam_i].tolist()):
            #     #         vis_bbox(bg, bbox, 384, (1920, 1080), clr=(0, 255, 0))
                
            #     pred, track_result = track_boxcolor(pred, pred_last, track_color, sample['orig_img'], last_img)
            #     last_img = sample['orig_img']

            #     #### use last 2d point to fix this box
            #     # for cam_i in range(views):
            #     #     H_point_min = (torch.min(last_point_org[cam_i][1, :])+420)//5  # 5=(1920//1080)
            #     #     last_box_p9 = (H_point_min - pred_last[cam_i][0,1])/(pred_last[cam_i][0,3]-pred_last[cam_i][0,1])
            #     #     print(cam_i, last_box_p9)
            #     #     box_p9 = (H_point_min - pred[cam_i][0,1])/(pred[cam_i][0,3]-pred[cam_i][0,1])
            #     #     # if box_p9 > last_box_p9*1.1:
            #     #     #     pred[cam_i][0,1] = H_point_min - last_box_p9*(pred[cam_i][0,3]-pred[cam_i][0,1])
            #     #     if box_p9 > 0.1:
            #     #         pred[cam_i][0,1] = H_point_min - 0.1*(pred[cam_i][0,3]-pred[cam_i][0,1])

            #     #### visulize video
                # for cam_i in range(len(pred)):
                #     # bg = sample['orig_img'][cam_i]
                #     vis_img = copy.deepcopy(sample['orig_img'][cam_i])
                #     # for i, bbox in enumerate(pred[cam_i].tolist()):
                #     #     vis_bbox(vis_img, bbox, int(args.inp_dim), (1920, 1080), clr=(255, 0, 0))
                #     #     vis_bbox(vis_img, pred_last[cam_i][0], int(args.inp_dim), (1920, 1080), clr=(0, 0, 255))

                #     for n in range(17):   # draw 2d point
                #         cor_x, cor_y = int(last_point_org[cam_i][0, n]), int(last_point_org[cam_i][1, n])
                #         # cv2.circle(bg, (cor_x, cor_y), 4, p_color[n], -1)
                #         cv2.circle(vis_img, (cor_x, cor_y), 7, (0, 0, 255), -1)
                    
                #     for c in connect:    # draw skeleton connect
                #         start_x, start_y = int(last_point_org[cam_i][0, c[0]]), int(last_point_org[cam_i][1, c[0]])
                #         end_x, end_y = int(last_point_org[cam_i][0, c[1]]), int(last_point_org[cam_i][1, c[1]])
                #         cv2.line(vis_img, (start_x, start_y), (end_x, end_y), (0,0,0), thickness=6)

                #     vis_img = cv2.resize(vis_img, (960, 540))
                #     cv2.imshow(str(cam_i), vis_img)
                #     cv2.waitKey(1)
                #     org_frames_save[f"{cam_i}_origimg"].append(
                #         cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))


            #     if track_result == 0:
            #         pred = copy.deepcopy(pred_last)
            #         # pred = kalmen_pred.copy()
            #         track_result = 1
            #         stop_track += 1
            #         if stop_track > 4:
            #             stop_track = 0
            #             track_result = 0
            #     else:
            #         pred_last = copy.deepcopy(pred)
            #         stop_track = 0

            # print("time track :", time.time() - aaa)

            torch.cuda.synchronize()
            local_times['after_detection'] = time.time() - local_times['start']
            boxes = []
            for i, det in enumerate(pred):  # per image
                seen += 1
                im0 = sample['orig_img'][i].copy()
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(sample['images'][i].shape[1:], det[:, :4], im0.shape).round()
                    boxes.append(det[:, :4])
            
            # local_times['after_detection'] = time.time() - local_times['start']

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
            # local_times['idk1'] = time.time() - local_times['start']

            last_box_org=[]
            for k in range(len(orig_img)):
                torch.cuda.synchronize()
                start = time.time()
                inp = orig_img[k] # cv2.cvtColor(orig_img[k], cv2.COLOR_BGR2RGB)
                image_shape = config.image_shape
                cameras_k = cameras[k]

                boxes_k = boxes[k].unsqueeze(0)
                box = tuple(np.array(boxes_k[0]))
                box = changeBox(box, inp)

                last_box_org.append(box)  # for fix the bbox for the next frame

                inp_s = crop_image(inp, box)
                # cv2.imwrite("{}.jpg".format(k), inp_s)
                # inp_s = crop_image_np(inp, box)
                cameras_k.update_after_crop(box)

                image_shape_before_resize = inp_s.shape[:2]
                inp_s = resize_image(inp_s, image_shape)
                # cv2.imwrite("{}.jpg".format(k), inp_s)
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

            # Pose Estimation
            images_batch, proj_matricies_batch = dataset_utils.prepare_batch_video(sample, views, device)

            torch.cuda.synchronize()
            local_times['before_3d'] = time.time() - local_times['start']

            keypoints_3d_pred, keypoints_2d_alg, heatmaps_alg, confidences_alg, lt_t1, lt_t2 = lt_model(images_batch,
                                                                                    proj_matricies_batch, sample)
            
            if k_i == 0 or str(sample['actname'])!=str(image_action_name_last): 
                save_keypoints_3d = keypoints_3d_pred
            else:
                save_keypoints_3d = torch.cat((save_keypoints_3d, keypoints_3d_pred), dim=0)
            save_3d_pred_point[''.join(sample['actname'])] = save_keypoints_3d.cpu().numpy()
            image_action_name_last = sample['actname']

            #### 2d point to org img
            for i, box_org in enumerate(last_box_org):
                xx = (box_org[0]+((keypoints_2d_alg[0,i,:,0])/384*(box_org[2]-box_org[0]))).unsqueeze(0)
                yy = (box_org[1]+((keypoints_2d_alg[0,i,:,1])/384*(box_org[3]-box_org[1]))).unsqueeze(0)
                last_point_org[i]=torch.concat([xx,yy], 0)

            # torch.cuda.synchronize()
            local_times['after_2d'] = lt_t1 - local_times['start']
            
            # torch.cuda.synchronize()
            local_times['after_3d'] = lt_t2 - local_times['start']

            # if k_i == 0:
            #     keypoints_3d = keypoints_3d_pred
            # else:
            #     # if k_i % 4 == 0:
            #     keypoints_3d = torch.cat((keypoints_3d, keypoints_3d_pred), dim=0)
            # torch.cuda.synchronize()
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

    # visualizer = read_data(keypoints_3d.cpu(), num_flame)

    # if args.smooth_weights != '' and os.path.isfile(
    #         args.smooth_weights):
    #     checkpoint = torch.load(args.smooth_weights)
    #     smooth_model.load_state_dict(checkpoint['state_dict'])
    #     print(f'==> Loaded pretrained model from {args.smooth_weights}...')
    # else:
    #     print(f'{args.smooth_weights} is not a pretrained model!!!!')
    #     exit()``

    # visualizer.visualize_3d(smooth_model)

    np.save('/home/zjlab/lizao/real-time-pose-estimation/pipeline3d_cam'+str(views)+'_pred.npy', save_3d_pred_point, allow_pickle=True)

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

    for l_i in org_frames_save:
        # size = (1080, 1920)
        fps = 30
        out_fn = os.path.join(temp_path, f"{str(l_i)}.avi")
        result = cv2.VideoWriter(out_fn,
                                cv2.VideoWriter_fourcc(*'DIVX'),
                                fps, (960, 540))
        for i, frame in enumerate(org_frames_save[l_i]):
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

    #     if l < 1:
    #         continue

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