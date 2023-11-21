# real time pose estimation

All resources can be found on the server: 10.101.104.54


The pre-trained model can be found at /data/users/yijia/learnable-triangulation-pytorch-master/videopose/yolo/yolov3-spp.weights

The input data can be found at /data/users/yijia/learnable-triangulation-pytorch-master/inputs/


1. Test the multi-videos then output the 3d keypoints
```bash
python video_test.py --eval --eval_dataset val --config experiments/human36m/eval/human36m_alg.yaml --logdir ./logs
```
If you want to test other videos, please change the paths of videos at videopose/gene_npz.py


=======
#### 介绍
{**以下是码云企业私有云说明，您可以替换此简介**
Gitee Premium 是码云企业级私有化部署方案，提供代码版本管理、项目管理、需求管理、缺陷管理、文档协作等功能，支持与企业内部 LDAP、项目管理、测试、部署、容器等平台的对接。}