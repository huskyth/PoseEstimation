import yaml
import cv2
import os
from collections import defaultdict
cv2.CAP_GSTREAMER

# ######save video as mp4, and change resolution
# videos = 22
# cams = 4
# for video in range(videos):
#     for cam in range(1,5):
#         cap_path = '/home/zjlab/dataset/video_23_906/V'+str(video)+'_0'+str(cam)+'.avi'
#         cap1 = cv2.VideoCapture(cap_path)
#         cap1.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc('M','J','P','G'))
#         frame_width1 = int(cap1.get(3))
#         frame_height1 = int(cap1.get(4))

#         SAVE_PATH = '/home/zjlab/dataset/video_23_906_mp4/V'+str(video)+'_0'+str(cam)+'.MP4'
#         out1 = cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (frame_width1, frame_height1))

#         print(cap1.get(cv2.CAP_PROP_FRAME_WIDTH), cv2.CAP_PROP_FRAME_WIDTH)

#         frames_to_save = defaultdict(list)
#         while True:
#             ret1, frame1 = cap1.read()
#             if ret1:
#                 frames_to_save[0].append(frame1)
#                 # cv2.imshow('frame1', frame1)
#                 if cv2.waitKey(1) & 0xFF == ord('1'):
#                     break
#             else:
#                 break
#         print("out the video: ", video, ", cam: ",  cam)

#         for i, frame in enumerate(frames_to_save[0]):
#             out1.write(frame)
#         cap1.release()
#         out1.release()
#         cv2.destroyAllWindows()

import numpy as np
import time
from tqdm import tqdm
# mask = np.load("/home/zjlab/Track-Anything/result/mask/V0_01/00100.npy")
# print(mask.shape)

# H, W= [], []
# time_begin = time.time()
# # for i in range(1080):
# #     for j in range(1920):
# #         if mask[i][j]!=0:
# #             W.append(i)
# #             H.append(j)
# # box = [min(H), min(W), max(H), max(W)]
# contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
# contours = contours[0] if len(contours) == 2 else contours[1]
# min_x, min_y = 1920, 1920
# max_x, max_y = 0, 0
# for cntr in contours:
#     x,y,w,h = cv2.boundingRect(cntr)
#     min_x = min(x, min_x)
#     min_y = min(y, min_y)
#     max_x = max(x+w, max_x)
#     max_y = max(y+h, max_y)
# box = [min_x, min_y, max_x, max_y]
# print(time.time()-time_begin)
# print(box)
# cap_path = '/home/zjlab/dataset/video_23_906/V'+str(0)+'_0'+str(1)+'.avi'
# cap1 = cv2.VideoCapture(cap_path)
# cap1.set(cv2.CAP_PROP_POS_FRAMES, 100)
# ret1, frame1 = cap1.read()
# cv2.rectangle(frame1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255,0,0), thickness=2)
# while(1):
#     cv2.imshow('frame1', frame1)
#     if cv2.waitKey(1) & 0xFF == ord('1'):
#         break

# video = 0
# cam = 1
# i = 0
# boxs = np.load('/home/zjlab/dataset/boxs_23_906/V'+str(video)+'_0'+str(cam)+'.npy')
# width = round(float((boxs[i][2] - boxs[i][0])/1920), 6)
# heigt = round(float((boxs[i][3] - boxs[i][1])/1080), 6)
# x = round(float((boxs[i][2] + boxs[i][0])/3840), 6)
# y = round(float((boxs[i][3] + boxs[i][1])/2160), 6)
# save_box = str(0)+' ' + str(x)+' ' + str(y)+' ' + str(width) +' ' + str(heigt)
# with open('/home/zjlab/dataset/yolo_fineturn/labels/train/'+str(1)+'.txt', 'w') as f:
#     f.write(save_box)





### train data v1-v10, v15-v17. val v18-v20, v21_1
save_num = 3496
save_vide0 = []
for video in range(21,22):
    for cam in range(1,2):
        cap_path = '/home/zjlab/dataset/video_23_906/V'+str(video)+'_0'+str(cam)+'.avi'
        cap1 = cv2.VideoCapture(cap_path)
        frame_width1 = int(cap1.get(3))
        frame_height1 = int(cap1.get(4))
        num_flame = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        print(num_flame)

        ##### save as yolov5 mode
        boxs = np.load('/home/zjlab/dataset/boxs_23_906/V'+str(video)+'_0'+str(cam)+'.npy')
        for i in range(num_flame):
            ### save label
            # width = round(float((boxs[i][2] - boxs[i][0])/frame_width1), 6)
            # heigt = round(float((boxs[i][3] - boxs[i][1])/frame_height1), 6)
            # x = round(float((boxs[i][2] + boxs[i][0])/(2*frame_width1)), 6)
            # y = round(float((boxs[i][3] + boxs[i][1])/(frame_height1*2)), 6)
            # save_box = str(0)+' ' + str(x)+' ' + str(y)+' ' + str(width) +' ' + str(heigt)
            # with open('/home/zjlab/dataset/yolo_fineturn/labels/val/'+str(save_num)+'.txt', 'w') as f:
            #     f.write(save_box)
            # ## save img
            # ret1, frame1 = cap1.read()
            # cv2.imwrite('/home/zjlab/dataset/yolo_fineturn/images/val/'+str(save_num)+'.jpg', frame1)
            # save_num = save_num + 1

            ## visualize
            with open('/home/zjlab/dataset/yolo_fineturn/labels/val/'+str(save_num)+'.txt','r', encoding='utf-8') as f:
                box_yolo = f.read().split(' ')
            print(type(box_yolo))
            print(box_yolo, box_yolo[1])
            x = float(box_yolo[1])*1920
            y = float(box_yolo[2])*1080
            width = float(box_yolo[3])*1920
            heigt = float(box_yolo[4])*1080
            box = [x-(width/2), y-(heigt/2), x+(width/2), y+(heigt/2)]
            print(box)
            cap_path = '/home/zjlab/dataset/yolo_fineturn/images/val/'+str(save_num)+'.jpg'
            frame1 = cv2.imread(cap_path, 1)
            cv2.rectangle(frame1, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0,255,0), thickness=2)
            cv2.circle(frame1, (int(x), int(y)), 7, (0, 0, 255), -1)
            save_num = save_num + 1
            cv2.imshow('frame1', frame1)
            cv2.waitKey(1)
print(save_num)

        ##### get boxs from track-anything mask
        # boxs=[]
        # print('V'+str(video)+'_0'+str(cam)+' start')
        # for flame in tqdm(range(num_flame)):
        #     time_begin = time.time()
        #     mask = np.load('/home/zjlab/Track-Anything/result/mask/V'+str(video)+'_0'+str(cam)+"/"+'{:05d}.npy'.format(flame))
        #     H, W= [], []
        #     # very slow, need to updata
        #     # for i in range(1080):
        #     #     for j in range(1920):
        #     #         if mask[i][j]!=0:
        #     #             W.append(i)
        #     #             H.append(j)
        #     # box = [min(H), min(W), max(H), max(W)]
        #     contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     contours = contours[0] if len(contours) == 2 else contours[1]
        #     min_x, min_y = 1920, 1920
        #     max_x, max_y = 0, 0
        #     for cntr in contours:
        #         x,y,w,h = cv2.boundingRect(cntr)
        #         min_x = min(x, min_x)
        #         min_y = min(y, min_y)
        #         max_x = max(x+w, max_x)
        #         max_y = max(y+h, max_y)
        #     box = [min_x, min_y, max_x, max_y]

        #     boxs.append(box)
        #     print('box:', box, 'flame:', flame, ' time:',time.time()-time_begin)
        # np.save('/home/zjlab/dataset/boxs_23_906/V'+str(video)+'_0'+str(cam)+'.npy', boxs)
        # print('V'+str(video)+'_0'+str(cam)+' finish')

        ####### visualize and save demo video
        # boxs = np.load('/home/zjlab/dataset/boxs_23_906/V'+str(video)+'_0'+str(cam)+'.npy')
        # SAVE_PATH = '/home/zjlab/dataset/video_23_906_mp4/demoV'+str(video)+'_0'+str(cam)+'.MP4'
        # out1 = cv2.VideoWriter(SAVE_PATH, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (frame_width1, frame_height1))
        # frames_to_save=[]
        # for j in range(num_flame):
        #     ret1, frame1 = cap1.read()
        #     cv2.rectangle(frame1, (int(boxs[j][0]), int(boxs[j][1])), (int(boxs[j][2]), int(boxs[j][3])), (0,255,0), thickness=2)
        #     cv2.imshow('frame1', frame1)
        #     frames_to_save.append(frame1)
        #     if cv2.waitKey(1) & 0xFF == ord('1'):
        #         break
        # for i, frame in enumerate(frames_to_save):
        #     out1.write(frame)
        # cap1.release()
        # out1.release()
        # cv2.destroyAllWindows()


