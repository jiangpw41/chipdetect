# 环境安装
该分支为main分支的windows版本，
主要增加了简单的UI界面基于Pyinstaller打包为exe文件的内容
```bash
conda create --name "chipdetect" python==3.9 -y
conda activate chipdetect

cd chipdetect
pip install -r requirements.txt
pip install -q git+https://github.com/THU-MIG/yolov10.git
```
# 项目结构

├── datasets         存放所有数据的文件夹

├── models           存放预训练模型权重和训练结果

├── tests            存放测试集表现和每次的测试得分

└── **config.json**      存放配置文件，核心，使用前必须查看并根据需要进行修改

└── default.yaml     yolov10超参数介绍

└── DataProcessor.py 数据预处理器windows版

└── TrainAndTest.py  训练验证测试器windows版

# 快速使用

拉取项目到本地后，执行上述**环境安装**步骤

在datasets目录下新建raw_datasets子目录，将原始数据集分文件保存到该目录下，并在config.json文件中根据数据集名称进行分组，所有分到一组的数据集将被整合到一个训练/测试集中使用同一个模型进行训练。

运行DataProcessor.py程序，对原始数据进行处理，获得统计数据。

在config.json文件中指定你要使用的yolo预训练模型（默认为yolov10n，首次运行会执行models/download_yolov10.sh脚本下载所有预训练权重）和要处理的数据集分组（默认为封口环空洞），执行TrainAndTest.py，获得训练结果和测试结果。训练结果保存在models/post-trained下。此外，你可以在config.json文件中train_parameters部分指定一些关键超参数，例如epoch, batch, GPU编号等。超参数作用详情见根目录下default.yaml文件。

# exe打包
激活chipdetect环境并进入项目根目录后，在cmd下执行
```cmd
pyinstall --onefile DataProcessor.py
pyinstall --onefile TrainAndTest.py
pyinstall --onefile ui.py
```