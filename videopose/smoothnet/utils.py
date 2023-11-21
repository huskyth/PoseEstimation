import os
import logging
from os import path as osp
import time
import yaml
import numpy as np
import torch


def create_logger(logdir, phase='train'):
    os.makedirs(logdir, exist_ok=True)
    # runs_loddir = logdir+"/runs"
    # os.makedirs(runs_loddir)

    log_file = osp.join(logdir, f'{phase}_log.txt')


    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=log_file, format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger


def save_dict_to_yaml(obj, filename, mode='w'):
    with open(filename, mode) as f:
        yaml.dump(obj, f, default_flow_style=False)


def prepare_output_dir(cfg, cfg_file):

    # ==== create logdir
    logtime = time.strftime('%d-%m-%Y_%H-%M-%S')
    logdir = f'{logtime}_{cfg.EXP_NAME}'

    logdir = osp.join(cfg.OUTPUT_DIR, logdir)
    
    dir_num=0
    logdir_tmp=logdir

    while os.path.exists(logdir_tmp):
        logdir_tmp = logdir + str(dir_num)
        dir_num+=1
    
    logdir=logdir_tmp
    
    os.makedirs(logdir, exist_ok=True)
    #shutil.copy(src=cfg_file, dst=osp.join(cfg.OUTPUT_DIR, 'config.yaml'))

    cfg.LOGDIR = logdir

    # save config
    save_dict_to_yaml(cfg, osp.join(cfg.LOGDIR, 'config.yaml'))

    return cfg


def worker_init_fn(worker_id):
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([worker_id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))

def slide_window_to_sequence(slide_window,window_step,window_size):  # 这是用了之前的数据转换为sequence，我们要写一个新的
    output_len=(slide_window.shape[0]-1)*window_step+window_size
    # output_len = slide_window.shape[0]
    sequence = [[] for i in range(output_len)]

    # 每个len(slide_window)里都有一个s[i],把每个s[i] append起来，然后做平均.sequence的维度应该是[out_len, window_size, 51]
    for i in range(slide_window.shape[0]):
        for j in range(window_size):
            sequence[i * window_step + j].append(slide_window[i, j, ...]) # ...表示后面的和原数据一样，没改变

    # 这里把sequence[out_len, window_size, 51]的window_size mean掉变为[out_len, 51]
    for i in range(output_len):
        sequence[i] = torch.stack(sequence[i]).type(torch.float32).mean(0)

    sequence = torch.stack(sequence)

    return sequence

def out2_to_sequence(slide_window,window_step,window_size):  # 这是用了之前的数据转换为sequence，我们要写一个新的
    output_len=slide_window.shape[0]  # 这里不能这样，要把维度变回去
    sequence = [[] for i in range(output_len)]

    # 每个len(slide_window)里都有一个s[i],把每个s[i] append起来，然后做平均.sequence的维度应该是[out_len, window_size, 51]
    for i in range(output_len):
        sequence[i] = slide_window[i, 0, ...].type(torch.float32)
    sequence = torch.stack(sequence)

    return sequence
#####下午测试这个！！！
# def window_to_seq_only_last(slide_window,window_step,window_size):  # 这是不使用之前的数据，数据变换哪里也要改
#     output_len=(slide_window.shape[0]-1)*window_step+window_size
#     sequence = [[] for i in range(output_len)]
#
#     # 每个len(slide_window)里都有一个s[i],把每个s[i] append起来，然后做平均.sequence的维度应该是[out_len, window_size, 51]
#     for i in range(slide_window.shape[0]):
#         for j in range(window_size):
#             sequence[i * window_step + j].append(slide_window[i, j, ...])
#
#     # sequence[0:window_size-1] = slide_window[0, :, :]
#     for i in range(output_len): # 不是range slide_window
#         if i < window_size:
#             sequence[i] = slide_window[0, i, ...]
#         else:
#             sequence[i] = slide_window[i - window_size + 1, -1, ...]
#             # sequence[i+window_size] = slide_window[i+1, -1, ...]
#         # sequence[i + window_size].append(slide_window[i + 1, -1, ...])
#
#     # # 这里把sequence[out_len, window_size, 51]的window_size mean掉变为[out_len, 51]
#     # for i in range(output_len):
#     #     sequence[i] = torch.stack(sequence[i]).type(torch.float32).mean(0)
#
#     sequence = torch.stack(sequence)
#
#     return sequence.type(torch.float32)

def window_to_seq_only_last(slide_window,window_size,mode):  # 这是不使用之前的数据，数据变换哪里也要改
    if mode == "mean":
        return slide_window.mean(1)
    elif mode == "last":
        return slide_window[:,window_size-1,:]
    elif mode == "last_pred":
        output_len = slide_window.shape[0] * 2
        sequence = [[] for j in range(output_len)]
        j = 0
        for i in range(slide_window.shape[0]):
            sequence[j] = slide_window[i, window_size - 1, :]
            j = j + 1
            sequence[j] = slide_window[i, window_size, :]
            j = j + 1
        return torch.stack(sequence)
    elif mode == "out2_pred":  # out2
        output_len = slide_window.shape[0] * 2
        sequence = [[] for j in range(output_len)]
        j = 0
        for i in range(slide_window.shape[0]):
            sequence[j] = slide_window[i, 0, :]
            j = j + 1
            sequence[j] = slide_window[i, 1, :]
            j = j + 1
        return torch.stack(sequence)
    elif mode == "out1_pred":  # out2
        output_len = slide_window.shape[0]
        sequence = [[] for j in range(output_len)]
        for i in range(slide_window.shape[0]):
            sequence[i] = slide_window[i, 1, :]
        return torch.stack(sequence)
    elif mode == "out2_mean":  # out2 使用了mean来达到fps翻倍
        output_len = slide_window.shape[0] * 2 - 2
        sequence = [[] for j in range(output_len)]
        j = 0
        for i in range(slide_window.shape[0]-1):
            sequence[j] = slide_window[i, 1, :]
            j = j + 1
            sequence[j] = (slide_window[i, 1, :]+slide_window[i+1, 1, :])/2
            j = j + 1
        return torch.stack(sequence)
    # 预测数值没用mean
    elif mode == "mean_pred":
        output_len = slide_window.shape[0] * 2
        sequence = [[] for j in range(output_len)]
        j = 0
        for i in range(slide_window.shape[0]):
            sequence[j] = slide_window[i, 0:window_size, :].mean(0)
            j = j+1
            sequence[j] = slide_window[i, window_size, :]
            j = j + 1
        return torch.stack(sequence)
    # 预测数值用了mean
    elif mode == "mean_pred_m":
        output_len = slide_window.shape[0] * 2
        sequence = [[] for j in range(output_len)]
        j = 0
        for i in range(slide_window.shape[0]):
            sequence[j] = slide_window[i, 0:window_size, :].mean(0)
            j = j+1
            sequence[j] = slide_window[i, window_size-1:window_size+1, :].mean(0)
            j = j + 1
        return torch.stack(sequence)
    else:
        AttributeError("no such mode")

