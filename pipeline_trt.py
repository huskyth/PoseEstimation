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
    parser.add_argument("--vid1", type=str, default="/home/zjlab/dataset/video_paper_10.30/video3/orig_img0.avi", help="Path to video from camera 1")
    parser.add_argument("--vid2", type=str, default="/home/zjlab/dataset/video_paper_10.30/video3/orig_img1.avi", help="Path to video from camera 2")
    parser.add_argument("--vid3", type=str, default='/home/zjlab/dataset/video_paper_10.30/video3/orig_img2.avi', help="Path to video from camera 3")
    parser.add_argument("--vid4", type=str, default='/home/zjlab/dataset/video_paper_10.30/video3/orig_img3.avi', help="Path to video from camera 4")
    parser.add_argument("--lbls", type=str, default="./human36m-multiview-labels-GTbboxes.npy", help="Path to labels with camera parameters")
    parser.add_argument('--conf-thres', type=float, default=0.3, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')

    parser.add_argument("--intri", type=str, default="/home/zjlab/calibration/EasyMocap-master-4/data/extri_data/intri.yml", help="Path to intri parameters")
    parser.add_argument("--extri", type=str, default="/home/zjlab/calibration/EasyMocap-master-4/data/extri_data/extri.yml", help="Path to extri parameters")

    parser.add_argument('--slide_window_size', type=int, default="32", help='slide window size')
    parser.add_argument('--smooth_weights', type=str, default="videopose/smoothnet/checkpoint.pth.tar",help='pretrained checkpoint file path')



    args = parser.parse_args()
    return args
save_name_3Dpose = 'person3_video7.npy'

def det_preproc(streams, camera_labels, inp_dim, cameras_all=None):    
    sample = defaultdict(list)

    for v in range(len(streams)):

        stream = streams[v]
        grabbed, frame = stream.read()
        print(frame.shape)

        # if the `grabbed` boolean is `False`, then we have
        # reached the end of the video file
        if not grabbed:
            print('===========================> This video get ' + str(k) + ' frames in total.')
            sys.stdout.flush()
            exit()

        # v = v+1

        retval_camera = Camera(cameras_all['0{}'.format(v+1)]['R'], cameras_all['0{}'.format(v+1)]['T'], cameras_all['0{}'.format(v+1)]['K'],
                            cameras_all['0{}'.format(v+1)]['dist'], str(v+1))
  
        # process and add the frame to the queue
        img_k, orig_img_k, im_dim_list_k = prep_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), inp_dim)

        sample['images'].append(img_k)
        sample['orig_img'].append(orig_img_k)
        sample['im_dim_list'].append(im_dim_list_k)
        sample['cameras'].append(retval_camera)
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
            predicted_pos = model(data_pred_window) # batch flame 51
            data_pred_window = data_pred_window.permute(0, 2, 1)
            predicted_pos = predicted_pos.permute(0, 2, 1)

        # 把数据还原为之前的大小
        mode2 = "out2_pred"  # 输出两倍fps
        predicted_pos2 = window_to_seq_only_last(predicted_pos, self.slide_window_size, mode=mode2).reshape(-1, keypoint_number, 3)
        print("out2 flame:",predicted_pos2.shape)

        save_name = mode2+"_3D.npy"

        np.save(os.path.join(temp_path, save_name), predicted_pos2.cpu().numpy())
        print("out2 name:",save_name)

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
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), clr, thickness=5)
    cv2.putText(img, str('%.2f'% conf), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color=clr)


if __name__ == '__main__':
    args = parse_args()

    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), 
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127)]
    connect = [(0,1),(1,2),(2,6),(5,4),(4,3),(3,6),(6,7),(7,8),(8,16),(9,16),(8,12),(11,12),(10,11),(8,13),(13,14),(14,15)]

    # temp_path = './temp_folder/demo_pipeline_4cam/'
    temp_path = '/home/zjlab/dataset/video_paper_10.30/video3/'
    os.makedirs(temp_path, exist_ok=True)
    frames_to_save = defaultdict(list)
    org_frames_save = defaultdict(list)

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

    videos_paths = [args.vid1, args.vid2, args.vid3, args.vid4]
    streams = []
    views = len(videos_paths)

    for p in videos_paths:
        stream = cv2.VideoCapture(p)
        streams.append(stream)
        assert stream.isOpened(), 'Cannot capture source'
    
    steam_len = int(streams[0].get(cv2.CAP_PROP_FRAME_COUNT))

    times['models_open_streams'] = [time.time() - times['global_start'][0]]

    camera_labels = None # np.load(args.lbls, allow_pickle=True).item()

    # times['models_load_lbls'] = [time.time() - times['global_start'][0]]
    track_result = 2
    pred_last = [torch.tensor([[210.62500, 135.00000, 256.50000, 274.00000,   0.90039,   0.00000]]),
                 torch.tensor([[216.62500, 138.50000, 255.62500, 270.50000,   0.82568,   0.00000]])]

    track_color = []
    last_img=[]
    stop_track = 0
    pred = []

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

    
    for k_i in tqdm(range(steam_len)):
    # for k_i in tqdm(range(2)):
        local_times = {}

        torch.cuda.synchronize()
        local_times['start'] = time.time()

        sample = det_preproc(streams, camera_labels, int(args.inp_dim), cameras_all=cameras_all)

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
            torch.cuda.synchronize()
            print("yolo time", time.time()-yolotime)
            #torch.cuda.synchronize()
            #local_times['dt1'] = time.time() - local_times['start']

            # prediction = prediction[0].reshape(views, -1, 85)
            

            #prediction = torch.from_numpy(prediction[0].reshape(views, -1, 85)).to(device)
            # pred = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0)
            aaa = time.time()
            if track_result==2:  # when change camera do it to updata pred_last
                # last_img = copy.deepcopy(sample['orig_img'])
                # track_result = 1
                pred_begin = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=100)
                for i in range(len(pred_begin)):
                    for j in range(len(pred_begin[i])):
                        if box_are(pred[i][0]) < box_are(pred_begin[i][j]):
                            pred[i][0] = pred_begin[i][j]
                    a = fast_color_histogram(sample['orig_img'][i], pred[i][0])
                    track_color.append(a)
                if box_are(pred[0][0])==0 or box_are(pred[1][0])==0:
                    track_result = 2
                else:
                    print(pred)
                    last_img = copy.deepcopy(sample['orig_img'])
                    pred_last = copy.deepcopy(pred)
                    track_result = 1

            if track_result == 0:
                pred_begin = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=100)
                for i, bboxs in enumerate(pred_begin):
                    track_id = compare_img(sample['orig_img'][i], list(bboxs), track_color[i])
                    if track_id==-1:
                        pred = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=1)
                        break
                    else:
                        pred[i][0] = bboxs[track_id]
                pred_last = copy.deepcopy(pred)
                track_result = 1


            else:
                ### kalmen filter maybe use for lose track
                # aaaa = time.time()
                # for i in range(views):
                #     kalmen_input[2 * i][0:2] = pred_last[i][0][0:2]
                #     kalmen_input[2 * i + 1][0:2] = pred_last[i][0][2:4]
                # kalmen_out = torch.Tensor(Kalman_Filtering_2D.predict(kalmen_input))
                # kalmen_pred[0][0] = kalmen_out[0:4]
                # kalmen_pred[1][0] = kalmen_out[4:8]
                # print("kalmen :", time.time() - aaaa)
                pred_save = []

                aaaa = time.time()
                pred = non_max_suppression(prediction, args.conf_thres, args.iou_thres, 0, max_det=100)
                nms_time = time.time()
                print("nms track :", nms_time - aaaa)
                pred_save = copy.deepcopy(pred)  # for visualize
                # for cam_i in range(len(pred)):
                #     bg = copy.deepcopy(sample['orig_img'][cam_i])
                #     for i, bbox in enumerate(pred[cam_i].tolist()):
                #         vis_bbox(bg, bbox, 384, (1920, 1080), clr=(0, 255, 0))
                    # org_frames_save[f"{cam_i}_yoloimg"].append(cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))

                    # org_frames_save[f"{cam_i}_yoloimg"].append(
                    #     cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
                
                pred, track_result = track_boxcolor(pred, pred_last, track_color, sample['orig_img'], last_img)
                last_img = sample['orig_img']

                #### use last 2d point to fix this box
                # for cam_i in range(views):
                #     H_point_min = (torch.min(last_point_org[cam_i][1, :])+420)//5  # 5=(1920//1080)
                #     last_box_p9 = (H_point_min - pred_last[cam_i][0,1])/(pred_last[cam_i][0,3]-pred_last[cam_i][0,1])
                #     print(cam_i, last_box_p9)
                #     box_p9 = (H_point_min - pred[cam_i][0,1])/(pred[cam_i][0,3]-pred[cam_i][0,1])
                #     # if box_p9 > last_box_p9*1.1:
                #     #     pred[cam_i][0,1] = H_point_min - last_box_p9*(pred[cam_i][0,3]-pred[cam_i][0,1])
                #     if box_p9 > 0.1:
                #         pred[cam_i][0,1] = H_point_min - 0.1*(pred[cam_i][0,3]-pred[cam_i][0,1])

                #### visulize video
                for cam_i in range(len(pred)):
                    vis_img = copy.deepcopy(sample['orig_img'][cam_i])
                    for i, bbox in enumerate(pred_save[cam_i].tolist()):  # vis boxs
                        vis_bbox(vis_img, bbox, 384, (1920, 1080), clr=(0, 255, 0))
                    for i, bbox in enumerate(pred[cam_i].tolist()):  # vis track box
                        vis_bbox(vis_img, bbox, int(args.inp_dim), (1920, 1080), clr=(255, 0, 0))
                        # vis_bbox(vis_img, pred_last[cam_i][0], int(args.inp_dim), (1920, 1080), clr=(0, 0, 255))  # vis last box
                    # org_frames_save[f"{cam_i}_trackimg"].append(cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

                    for c in connect:    # draw skeleton connect
                        start_x, start_y = int(last_point_org[cam_i][0, c[0]]), int(last_point_org[cam_i][1, c[0]])
                        end_x, end_y = int(last_point_org[cam_i][0, c[1]]), int(last_point_org[cam_i][1, c[1]])
                        cv2.line(vis_img, (start_x, start_y), (end_x, end_y), (255,0,0), thickness=6)
                    for n in range(17):   # draw 2d point
                        cor_x, cor_y = int(last_point_org[cam_i][0, n]), int(last_point_org[cam_i][1, n])
                        # cv2.circle(bg, (cor_x, cor_y), 4, p_color[n], -1)
                        cv2.circle(vis_img, (cor_x, cor_y), 7, (0, 0, 255), -1)

                    # vis_img = cv2.resize(vis_img, (960, 540))
                    cv2.imshow(str(cam_i), vis_img)
                    cv2.waitKey(1)
                    # org_frames_save[f"{cam_i}_allimg"].append(cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))


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

            print("time track :", time.time() - nms_time)

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
            images_batch, proj_matricies_batch = dataset_utils.prepare_batch_video(sample, len(videos_paths), device)

            torch.cuda.synchronize()
            local_times['before_3d'] = time.time() - local_times['start']

            keypoints_3d_pred, keypoints_2d_alg, heatmaps_alg, confidences_alg, lt_t1, lt_t2 = lt_model(images_batch,
                                                                                    proj_matricies_batch, sample)
            #### 2d point to org img
            for i, box_org in enumerate(last_box_org):
                xx = (box_org[0]+((keypoints_2d_alg[0,i,:,0])/384*(box_org[2]-box_org[0]))).unsqueeze(0)
                yy = (box_org[1]+((keypoints_2d_alg[0,i,:,1])/384*(box_org[3]-box_org[1]))).unsqueeze(0)
                last_point_org[i]=torch.concat([xx,yy], 0)

            # torch.cuda.synchronize()
            local_times['after_2d'] = lt_t1 - local_times['start']
            
            # torch.cuda.synchronize()
            local_times['after_3d'] = lt_t2 - local_times['start']

            if k_i == 0:
                keypoints_3d = keypoints_3d_pred
            else:
                # if k_i % 4 == 0:
                keypoints_3d = torch.cat((keypoints_3d, keypoints_3d_pred), dim=0)
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
    #     exit()

    # visualizer.visualize_3d(smooth_model)


    np.save(os.path.join(temp_path, '3D-Pose.npy'), keypoints_3d.cpu(), allow_pickle=True)
    # np.save(os.path.join("/home/zjlab/dataset/hospital10-13/10_13_person3/3D-pose", save_name_3Dpose), keypoints_3d.cpu(), allow_pickle=True)
    
    # print(keypoints_3d.shape)

    # for l_i in frames_to_save:
    #     size = (config.image_shape[0], config.image_shape[1])
    #     fps = 50
    #     out_fn = os.path.join(temp_path, f"{str(l_i)}.avi")
    #     result = cv2.VideoWriter(out_fn,
    #                             cv2.VideoWriter_fourcc(*'DIVX'),
    #                             fps, size)
    #     for i, frame in enumerate(frames_to_save[l_i]):
    #         result.write(frame)
    #     result.release()

    for l_i in org_frames_save:
        # size = (1920, 1080)
        fps = 15
        out_fn = os.path.join(temp_path, f"{str(l_i)}.avi")
        result = cv2.VideoWriter(out_fn,
                                cv2.VideoWriter_fourcc(*'DIVX'),
                                fps, (1920, 1080))
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