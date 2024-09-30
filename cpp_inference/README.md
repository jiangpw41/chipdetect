# 概述
YOLOv10的C++格式推理已有开源项目https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference

# 项目介绍
项目特点如下：
- 目前只在Ubuntu18.04, 20.04, 22.04三个型号上测试过，其他系统要使用得通过虚拟化工具，如Windows使用WSL或虚拟机
- 支持手动部署和docker部署
- 支持三种识别模式
  - 读取本地视频实时显示：可行
  - 读取本地单张图片并实时显示：已经部署并实践成功，后续如果需要对本地图片进行批量处理并保存结果仅需需改部分代码。
  - 读取摄像头视频流并实时显示：困难较大，非Ubuntu系统使用虚拟化技术的难点在于摄像头的映射，windows在WSL使用摄像头参见https://blog.csdn.net/breaksoftware/article/details/140752840

# 部署指令

## Docker部署环境
较为简单，只需要拉取远程镜像并进入
```bash
docker pull rlggyp/yolov10:20.04
sudo apt install x11-xserver-utils
xhost +local:docker    # 允许 Docker 容器内的应用程序访问宿主机的 X11 服务器（即图形界面），服务器没有图形界面就不执行
docker run -it --rm --mount type=bind,src=$(pwd),dst=/repo --env DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v /dev:/dev -w /repo rlggyp/yolov10:20.04
```

## 手动部署环境

```bash
sudo apt-get update
sudo apt-get install -y \
    libtbb2 \
    cmake \
    make \
    git \
    libyaml-cpp-dev \
    wget \
    libopencv-dev \
    pkg-config \
    g++ \
    gcc \
    libc6-dev \
    make \
    build-essential \
    sudo \
    ocl-icd-libopencl1 \
    python3 \
    python3-venv \
    python3-pip \
    libpython3.8

wget -O openvino.tgz https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.3/linux/l_openvino_toolkit_ubuntu20_2023.3.0.13775.ceeafaf64f3_x86_64.tgz && \
sudo mkdir /opt/intel
sudo mv openvino.tgz /opt/intel/
cd /opt/intel
sudo tar -xvf openvino.tgz
sudo rm openvino.tgz
sudo mv l_openvino* openvino
```

## 上述两者共用的后续安装
```bash
git clone https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference.git
cd YOLOv10-OpenVINO-CPP-Inference/src

mkdir build
cd build
cmake
make

# 编译完毕后，回到repo目录，创建权重保存文件，下面演示的是从远程直接下载预训练的onnx权重加载，也可以设置为加载本地训练结果
cd /repo
mkdir weights
cd /repo/weights
wget https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference/raw/model/assets/yolov10n.onnx
wget https://github.com/rlggyp/YOLOv10-OpenVINO-CPP-Inference/raw/model/assets/yolov10n_int8_openvino.zip
sudo apt-get update
sudo apt-get install unzip
unzip yolov10n_int8_openvino.zip
mv yolov10n_int8_openvino/metadata.yaml .
rm -r yolov10n_int8_openvino
rm yolov10n_int8_openvino.zip

cd /repo/src/build
./detect /repo/weights/yolov10n.onnx /repo/assets/bus.jpg   # 这个只需要有图形界面即可
./camera /repo/weights/yolov10n.onnx 0                      # 这一行如果在虚拟设备中，可能会因为摄像头设备映射问题无法执行
```