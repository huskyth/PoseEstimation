import numpy as np
from collections import defaultdict
import torch
import math
from mvn.utils.multiview import Camera
from torch.utils.data import Dataset
import os
import cv2
import json

### 把3d点转为json文件
def from_3d_to_json(val_3d_path, filename):
    kpt_s = ['Hip', 'RightHip', 'RightKnee', 'RightAnkle', 'LeftHip', 'LeftKnee', 'LeftAnkle',
                'Spine', 'Thorax', 'Neck', 'Head', 'LeftShoulder', 'LeftElbow', 'LeftWrist', 'RightShoulder',
                'RightElbow', 'RightWrist']
    val_3d_outputt = np.load(val_3d_path, allow_pickle=True)
    filename = filename
    point_change = [6, 2, 1, 0, 3, 4, 5, 7, 8, 16, 9, 13, 14, 15, 12, 11, 10]
    save_json = []
    print(val_3d_outputt.shape)
    for i, predict_pos in enumerate(val_3d_outputt):
        kpts_all = {joint: [] for joint in kpt_s}
        point3d = np.array(predict_pos[point_change, :])
        for j in range(len(point3d)):

            X = point3d[j][0]
            Y = point3d[j][1]
            Z = point3d[j][2]

            point3d[j][0] = -X
            point3d[j][1] = Z
            point3d[j][2] = -Y
        if i == 0:
            bone_len = np.linalg.norm(point3d[2] - point3d[3])
        unity_len = 0.43
        # unity_len = 0.56
        scale = unity_len / bone_len
        for j in range(len(point3d)):
            kpts_dict = []
            kpts_dict.append(float(point3d[j][0]*scale))
            kpts_dict.append(float(point3d[j][1]*scale))
            kpts_dict.append(float(point3d[j][2]*scale))
            kpts_all[kpt_s[j]] = kpts_dict
        save_json.append(kpts_all)
    with open (filename,'w') as f:
        msg = json.dump(save_json, f)

val_3d_path = '/home/zjlab/dataset/video_paper_10.30/video5/3D-Pose.npy'
filename = '/home/zjlab/dataset/video_paper_10.30/video5/3D-Pose.json'
# from_3d_to_json(val_3d_path, filename)


demo_path = "/home/zjlab/dataset/video_paper_10.30/video5/demo5.mov"
# demo_path = "/home/zjlab/dataset/video_paper_10.30/images1/0_skeimg"
demo = cv2.VideoCapture(demo_path)
demo_len = int(demo.get(cv2.CAP_PROP_FRAME_COUNT))

# img_num0 = [i for i in range(104,129,3)]
# img_num1 = [j for j in range(203,228,3)]
img_num1 = [j for j in range(202,334,14)]
print(img_num1)
for frame_index in range(demo_len):
    ret, frame = demo.read()
    if frame_index in img_num1:
        # frame = frame[210:480, 210:690]
        cv2.imwrite('/home/zjlab/dataset/video_paper_10.30/images8/paper_human/000'+str(frame_index)+'.png',frame)

demo.release()
cv2.destroyAllWindows()


# pose_path = "/home/zjlab/dataset/video_paper_10.30/video3/3D-Pose.avi"
# org_video = "/home/zjlab/dataset/video_paper_10.30/video3/orig_img0.avi"
# ske_video = "/home/zjlab/dataset/video_paper_10.30/video3/ske_img0.avi"
# videos_paths = [pose_path, org_video, ske_video]
# streams = []
# views = len(videos_paths)
# for p in videos_paths:
#     stream = cv2.VideoCapture(p)
#     streams.append(stream)
#     assert stream.isOpened(), 'Cannot capture source'
# steam_len = int(streams[0].get(cv2.CAP_PROP_FRAME_COUNT))
# print(steam_len)

# for frame_index in range(steam_len):
#     for stream in streams:
#         ret, frame = stream.read()
#     if frame_index % 30 != 0:  
#         frame_index += 1
#         for stream in streams:
#             ret, frame = stream.read()
#         continue
#     else:
#         for stream in streams:
#             ret, frame = stream.read()
#             cv2.imwrite('messigray.png',img)

#     frame_index += 1

# for video in streams:
#     video.release()
# cv2.destroyAllWindows()
















# ##计算两个3d点的mpjpe
# val_3d_outputt = np.load('/home/zjlab/lizao/real-time-pose-estimation/val_3d_output.npy', allow_pickle=True).item()  # only resnet
# val_3d_outputt = np.load('/home/zjlab/lizao/real-time-pose-estimation/pipeline3d_cam4_pred.npy', allow_pickle=True).item()  # our pipeline
# val_3d_gtt = np.load('/home/zjlab/lizao/real-time-pose-estimation/val_3d_gt.npy', allow_pickle=True).item()
# # get mpjpe
# def mpjpe(predicted, target):
#     """
#     Mean per-joint position error (i.e. mean Euclidean distance),
#     often referred to as "Protocol #1" in many papers.
#     """

#     assert predicted.shape == target.shape
#     return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

# def p_mpjpe(predicted, target):
#     """
#     Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
#     often referred to as "Protocol #2" in many papers.
#     """
#     assert predicted.shape == target.shape
    
#     muX = np.mean(target, axis=1, keepdims=True)
#     muY = np.mean(predicted, axis=1, keepdims=True)
    
#     X0 = target - muX
#     Y0 = predicted - muY

#     normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
#     normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
#     X0 /= normX
#     Y0 /= normY

#     H = np.matmul(X0.transpose(0, 2, 1), Y0)
#     U, s, Vt = np.linalg.svd(H)
#     V = Vt.transpose(0, 2, 1)
#     R = np.matmul(V, U.transpose(0, 2, 1))

#     # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
#     sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
#     V[:, :, -1] *= sign_detR
#     s[:, -1] *= sign_detR.flatten()
#     R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

#     tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

#     a = tr * normX / normY # Scale
#     t = muX - a*np.matmul(muY, R) # Translation
    
#     # Perform rigid transformation on the input
#     predicted_aligned = a*np.matmul(predicted, R) + t
    
#     # Return MPJPE
#     return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))

# def n_mpjpe(predicted, target):
#     """
#     Normalized MPJPE (scale only), adapted from:
#     https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
#     """
#     assert predicted.shape == target.shape
    
#     norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
#     norm_target = torch.mean(torch.sum(target*predicted, dim=3, keepdim=True), dim=2, keepdim=True)
#     scale = norm_target / norm_predicted
#     return mpjpe(scale * predicted, target)

# # 得到所有动作的测试结果
# def evl(val_3d_outputt, val_3d_gtt):
#     actions=[]
#     avg_e1 = 0
#     avg_e2 = 0
#     for key in val_3d_outputt:
#         key = key.split('_')
#         key = key[1].split('-')
#         if key[0] not in actions:
#             actions.append(key[0])
#     print(actions)

#     for action in actions:
#         epoch_loss_3d_pos_scale = 0
#         epoch_loss_3d_pos = 0
#         epoch_loss_3d_pos_procrustes = 0
#         N = 0
#         for key in val_3d_outputt:
#             if action in key:
#                 val_3d_output = torch.from_numpy(val_3d_outputt[key])
#                 val_3d_gt = torch.from_numpy(val_3d_gtt[key])

#                 error = mpjpe(val_3d_output, val_3d_gt)
#                 # epoch_loss_3d_pos_scale += val_3d_gt.shape[0]*val_3d_gt.shape[1] * n_mpjpe(val_3d_output, val_3d_gt).item()

#                 epoch_loss_3d_pos += val_3d_gt.shape[0]*val_3d_gt.shape[1] * error.item()
#                 N += val_3d_gt.shape[0] * val_3d_gt.shape[1]
                
#                 inputs = val_3d_gt.cpu().numpy().reshape(-1, val_3d_gt.shape[-2], val_3d_gt.shape[-1])
#                 val_3d_output = val_3d_output.cpu().numpy().reshape(-1, val_3d_gt.shape[-2], val_3d_gt.shape[-1])

#                 epoch_loss_3d_pos_procrustes += val_3d_gt.shape[0]*val_3d_gt.shape[1] * p_mpjpe(val_3d_output, inputs)

#                     # # Compute velocity error
#                     # epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)
                    
#         if action is None:
#             print('----------')
#         else:
#             print('----'+action+'----')
#         e1 = (epoch_loss_3d_pos / N)
#         e2 = (epoch_loss_3d_pos_procrustes / N)
#         # e3 = (epoch_loss_3d_pos_scale / N)*1000

#         print('Protocol #1 Error (MPJPE):', e1, 'mm')
#         print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
#         # print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
#         print('----------')
#         avg_e1 += e1
#         avg_e2 += e2
#     print('Protocol #1 avg Error (MPJPE):', avg_e1/len(actions), 'mm')
#     print('Protocol #2 avg Error (P-MPJPE):', avg_e2/len(actions), 'mm')

# evl(val_3d_outputt, val_3d_gtt)

### for train smoothnet
# smooth_data = defaultdict(list)
# smooth_train = []
# smooth_evl = []
# smooth_train_gt = []
# smooth_evl_gt = []
# for key in val_3d_output:
#     if 'S9' in key:
#         smooth_train.append(val_3d_output[key])
#         smooth_train_gt.append(val_3d_gt[key])
#         print('the train key is: ', key)

#     else:
#         if '-1' in key:
#             smooth_train.append(val_3d_output[key])
#             smooth_train_gt.append(val_3d_gt[key])
#             print('the train key is: ', key)

#         else:
#             smooth_evl.append(val_3d_output[key])
#             smooth_evl_gt.append(val_3d_gt[key])
#             print('the evl is: ', key)
# smooth_data['smooth_train']=smooth_train
# smooth_data['smooth_evl']=smooth_evl
# smooth_data['smooth_train_gt']=smooth_train_gt
# smooth_data['smooth_evl_gt']=smooth_evl_gt
# np.save('smooth_data', smooth_data)


