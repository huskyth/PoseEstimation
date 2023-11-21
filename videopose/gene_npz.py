import ntpath
import os
import shutil
import time

import numpy as np
import torch.utils.data
from tqdm import tqdm

import cv2

# from common.utils import calculate_area
from .dataloader import DetectionLoader, DetectionProcessor, VideoLoader
from mvn.datasets import utils as dataset_utils
from .fn import getTime


# from opt import opt
# from pPose_nms import write_json


def generate_kpts(args, model):
    final_result = handle_video(args, model)

    # ============ Changing ++++++++++

    # kpts = []
    # no_person = []
    # for i in range(len(final_result)):
    #     if not final_result[i]['result']:  # No people
    #         no_person.append(i)
    #         kpts.append(None)
    #         continue

    # kpt = max(final_result[i]['result'],
    #           key=lambda x: x['proposal_score'].data[0] * calculate_area(x['keypoints']), )['keypoints']

    # kpts.append(kpt.data.numpy())

    # for n in no_person:
    #     kpts[n] = kpts[-1]
    # no_person.clear()

    # for n in no_person:
    #     kpts[n] = kpts[-1]

    # ============ Changing End ++++++++++

    # name = f'{args.outputpath}/{video_name}.npz'
    # kpts = np.array(kpts).astype(np.float32)
    # print('kpts npz save in ', name)
    # np.savez_compressed(name, kpts=kpts)

    return final_result


def handle_video(args, model):
    path1 = '/root/repos/real-time-pose-estimation/inputs/Directions 1.54138969.mp4'
    path2 = '/root/repos/real-time-pose-estimation/inputs/Directions 1.55011271.mp4'
    path3 = '/root/repos/real-time-pose-estimation/inputs/Directions 1.58860488.mp4'
    path4 = '/root/repos/real-time-pose-estimation/inputs/Directions 1.60457274.mp4'
    videofile = [path1, path2, path3, path4]
    # videofile = [path1, path2]
    # mode = args.mode
    if not len(videofile):
        raise IOError('Error: must contain --video')
    # Load input video
    data_loader = VideoLoader(videofile, args, batchSize=1).start()
    (fourcc, fps, frameSize) = data_loader.videoinfo()
    print('the video is {} f/s'.format(fps))
    # =========== end video ===============
    # Load detection loader
    print('Loading YOLO model..')
    # sys.stdout.flush()
    det_loader = DetectionLoader(data_loader, args, batchSize=1).start()
    #  start a thread to read frames from the file video stream
    det_processor = DetectionProcessor(det_loader, args).start()
    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }
    print('Start pose estimation...')
    im_names_desc = tqdm(range(data_loader.length()))
    model.eval()

    p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0), 
               (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
               (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127)]
    for i in im_names_desc:

        start_time = getTime()
        with torch.no_grad():
            sample = det_processor.read()
            orig_img = sample['orig_img']
            boxes = sample['boxes_k']
            inp = sample['inp']
            if orig_img is None:
                print(f'{i}-th image read None: handle_video')
                break
            if boxes is None:
                continue

            # Pose Estimation
            images_batch, proj_matricies_batch = dataset_utils.prepare_batch_video(sample, len(videofile), 0)

            keypoints_3d_pred, keypoints_2d_alg, heatmaps_alg, confidences_alg = model(images_batch,
                                                                                       proj_matricies_batch, sample)
            key_2ds = keypoints_2d_alg[0]
            for v in range(len(key_2ds)):
                key_2d = key_2ds[v]
                for n in range(17):
                    cor_x, cor_y = int(key_2d[n][0]), int(key_2d[n][1])
                    bg = inp[v]
                    cv2.circle(bg, (cor_x, cor_y), 4, p_color[n], -1)
                # cv2.imwrite("/data/users/yijia/learnable-triangulation-pytorch-master/img_{}.jpg".format(v), bg)

            if i == 0:
                keypoints_3d = keypoints_3d_pred

            else:
                keypoints_3d = torch.cat((keypoints_3d, keypoints_3d_pred), dim=0)

            ckpt_time, post_time = getTime(start_time)
            runtime_profile['pn'].append(post_time)

        if args.profile:
            # TQDM
            im_names_desc.set_description(
                'post processing: {pn:.4f}'.format(pn=np.mean(runtime_profile['pn']))
            )

    np.save('outputs/alpha_result/test_3d_output.npy', keypoints_3d.cpu(), allow_pickle=True)
    print(keypoints_3d)

    return keypoints_3d


if __name__ == "__main__":
    os.chdir('../..')
    print(os.getcwd())

    # handle_video(img_path='outputs/image/kobe')
    generate_kpts('outputs/dance.mp4')
