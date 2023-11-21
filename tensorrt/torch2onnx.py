import torch
import onnx
import numpy as np
import argparse
import sys
import os

path = './mvn'
if path not in sys.path:
    sys.path.insert(0, path)

from models.pose_resnet import get_pose_net
from utils import cfg
# import onnxruntime as ort

batch_size = 4
def convert(model,path): 
    # # set the model to inference mode 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(batch_size, 3, 384, 384, requires_grad=True)  
    print(dummy_input.dtype)

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         path,       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=11,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['input'],   # the model's input names 
         output_names = ['heatmaps', 'features', 'alg_confidences'], # the model's output names 
        #  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes 
        #                         'heatmaps' : {0 : 'batch_size'}, 'features' : {0 : 'batch_size'}}
                                ) 
    print(" ") 
    print('Model has been converted to ONNX') 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='experiments/human36m/eval/human36m_alg.yaml', help="Path, where config file is stored")
    parser.add_argument("--pretrained_path", type=str, default="./data/pretrained/human36m/human36m_alg_10-04-2019/checkpoints/0060/weights.pth", help="Path, where config file is stored")
    parser.add_argument("--onnx_path", type=str, default='/home/zjlab/real-time-pose-estimation/data/pretrained/human36m/resnet152_4_384.onnx', help="Path, where config file is stored")
    parser.add_argument("--engine_path", type=str, default="/home/zjlab/real-time-pose-estimation/data/pretrained/human36m/resnet152_4_384.engine", help="Path, where config file is stored")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__": 
    args = parse_args()
    config = cfg.load_config(args.config)
    config.model.backbone.alg_confidences = True
    config.model.backbone.vol_confidences = False

    # # Load the torch model parameter
    model = get_pose_net(config.model.backbone, device='cuda:0')
    
    state_dict = torch.load(args.pretrained_path)
    for key in list(state_dict.keys()):
        new_key = key.replace("backbone.", "")
        state_dict[new_key] = state_dict.pop(key)
    model.load_state_dict(state_dict, strict=True)
    print("Successfully loaded pretrained weights for whole model")

    model.eval()
    convert(model, args.onnx_path)   # Convert torch to ONNX

    inputTensor = np.ones((batch_size, 3, 384, 384)).astype(np.float32)
    # # Test whether the torch and onnx outputs are consistent
    # model_onnx = onnx.load(args.onnx_path) 
    # onnx.checker.check_model(model_onnx)

    # session = ort.InferenceSession(args.onnx_path, providers=['CUDAExecutionProvider'])
    # onnx_outputs = session.run(None, {"input": inputTensor})
    # print("onnx out shape", onnx_outputs[0].shape, onnx_outputs[1].shape, onnx_outputs[2].shape)
    
    # with torch.no_grad():
    #     torch_out = model(torch.tensor(inputTensor))
    # # print("torch out shape", torch_out[0].shape, torch_out[1].shape, torch_out[2].shape)
    # # np.testing.assert_allclose(torch_out[0].cpu().numpy(), onnx_outputs[0], rtol=1e-03, atol=1e-05)
    # # np.testing.assert_allclose(torch_out[1].cpu().numpy(), onnx_outputs[1], rtol=1e-03, atol=1e-05)
    # # np.testing.assert_allclose(torch_out[2].cpu().numpy(), onnx_outputs[2], rtol=1e-03, atol=1e-05)
    # # print("onnx out is the same as torch")


    # #### Call the tensorRT model, just replace it with the code below
    # from onnx2tensorRT import TrtModel
    # TRT_model = TrtModel(args.engine_path)

    # trt_out = TRT_model(np.array(inputTensor), batch_size)
    # heatmaps = trt_out[2].reshape(batch_size, -1, 96, 96)
    # features = trt_out[1].reshape(batch_size, -1, 96, 96)
    # alg_confidences = trt_out[0].reshape(batch_size, -1)
    # print("tensorRT out shape", heatmaps.shape, features.shape, alg_confidences.shape)
    # ############

    # ### Test whether the tensorRT and onnx outputs are consistent
    # np.testing.assert_allclose(torch_out[0].cpu().numpy(), heatmaps, rtol=1e-03, atol=1e-05)
    # np.testing.assert_allclose(torch_out[1].cpu().numpy(), features, rtol=1e-03, atol=1e-05)
    # np.testing.assert_allclose(torch_out[2].cpu().numpy(), alg_confidences, rtol=1e-03, atol=1e-05)
    # print("tensoeRT out is the same as onnx")