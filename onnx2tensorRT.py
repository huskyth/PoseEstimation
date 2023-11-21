import os
import random
import sys
import numpy as np
import tensorrt as trt
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
import torch
import pycuda.autoinit

class ModelData(object):
    MODEL_PATH = "/data/users/lizao/SmoothNet/code/enterprise-real-time-pose-estimation-ps-wl/resnet152.onnx"
    INPUT_SHAPE = (1, 3, 384, 384)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


# You can set the logger severity higher to suppress messages (or lower to display more messages).
engine_model_file="/data/users/lizao/SmoothNet/code/enterprise-real-time-pose-estimation-ps-wl/resnet152.engine"


# onnx to engine，This part has errors and can be converted directly with trtexec.
def build_engine_onnx(model_file):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(ModelData.MODEL_PATH, "rb") as model:
        print("begin onnx file parsing")
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
        
    # 构建配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)

    print("start build_serialized_network")
    serialized_engine = builder.build_serialized_network(network, config)  # 这一步骤总是说数组溢出，换成trtexec可以转换
    print("Complete build_serialized_network")
    # 保存引擎
    with open(model_file, "wb") as f:
        f.write(serialized_engine)
    print("successful creat engine save: ", model_file)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
# # 读取tensorRT模型并推理。Read the tensorRT model and inference.
class TrtModel:
    
    def __init__(self,engine_path,max_batch_size=1,dtype=np.float32):
        
        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")             
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        
        return inputs, outputs, bindings, stream
              
    def __call__(self,x:np.ndarray,batch_size=2):
        
        x = x.astype(self.dtype)
        
        np.copyto(self.inputs[0].host,x.ravel())
        
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)
        
        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream) 
        
        self.stream.synchronize()
        return [out.host.reshape(batch_size,-1) for out in self.outputs]

def main():
    # inference TensorRT engine
    batch_size = 2
    model = TrtModel(engine_model_file)

    shape = model.engine.get_binding_shape(0)
    print(shape)
    data = np.ones((batch_size, 3, 384, 384)).astype(np.float32)
    trt_out = model(data,batch_size)
    print(len(trt_out))
    for i in range(len(trt_out)):
        print("trt_out", i, ":", trt_out[i].shape)
    trt_out[0] = trt_out[0].reshape(batch_size, -1)
    trt_out[1] = trt_out[1].reshape(batch_size, -1, 96, 96)
    trt_out[2] = trt_out[2].reshape(batch_size, -1, 96, 96)
    for i in range(len(trt_out)):
        print("after reshape trt_out", i, ":", trt_out[i].shape)

from collections import OrderedDict, namedtuple
class TrtModel_yolo:
    def __init__(self,engine_path, device='cuda:0'):    
        self.fp16 = False
        self.dynamic = False
        self.device = torch.device(device)
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)

        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())

        self.input_name = self.model.get_binding_name(0)
        self.context = self.model.create_execution_context()

        self.bindings = OrderedDict()
        self.output_names = []
        # self.cfx = cuda.Device(0).make_context()
        # cuda.init()

        print(self.input_name)
        for i in range(self.model.num_bindings):
            name = self.model.get_binding_name(i)

            dtype = trt.nptype(self.model.get_binding_dtype(i))
            if self.model.binding_is_input(i):
                if -1 in tuple(self.model.get_binding_shape(i)):  # dynamic
                    self.dynamic = True
                    self.context.set_binding_shape(i, tuple(self.model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    self.fp16 = True
            else:  # output
                self.output_names.append(name)
            shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())   
        self.batch_size = self.bindings[self.input_name].shape[0]
              
    def __call__(self,im):
        # self.cfx.push()
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()       
        if self.dynamic and im.shape != self.bindings[self.input_name].shape:
            i = self.model.get_binding_index(self.input_name)
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.bindings[self.input_name] = self.bindings[self.input_name]._replace(shape=im.shape)
            for name in self.output_names:
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
        s = self.bindings[self.input_name].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs[self.input_name] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        # self.cfx.pop()
        return [self.bindings[x].data for x in sorted(self.output_names)]


class TrtModel_yolo2:
    def __init__(self, engine_path, device='cuda:0'):
        self.fp16 = False
        self.dynamic = False
        self.device = torch.device(device)
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)

        with open(engine_path, 'rb') as f, trt.Runtime(self.logger) as runtime:
            self.model = runtime.deserialize_cuda_engine(f.read())

        self.context = self.model.create_execution_context()

        self.bindings = OrderedDict()
        self.output_names = []
        # self.cfx = cuda.Device(0).make_context()
        # cuda.init()

        for i in range(self.model.num_bindings):
            name = self.model.get_binding_name(i)

            dtype = trt.nptype(self.model.get_binding_dtype(i))
            if self.model.binding_is_input(i):
                if -1 in tuple(self.model.get_binding_shape(i)):  # dynamic
                    self.dynamic = True
                    self.context.set_binding_shape(i, tuple(self.model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    self.fp16 = True
            else:  # output
                self.output_names.append(name)
            shape = tuple(self.context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            self.bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.batch_size = self.bindings['images'].shape[0]

    def __call__(self, im):
        # self.cfx.push()
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()
        if self.dynamic and im.shape != self.bindings['images'].shape:
            i = self.model.get_binding_index('images')
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.bindings['images'] = self.bindings['images']._replace(shape=im.shape)
            for name in self.output_names:
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
        s = self.bindings['images'].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs['images'] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        # self.cfx.pop()
        return [self.bindings[x].data for x in sorted(self.output_names)]


if __name__ == "__main__":
    main()