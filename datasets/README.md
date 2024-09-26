# 文件夹介绍
本文件夹存放训练数据，结构如下

├── label_statistics.json   存放预处理过程中记录的数据集统计信息，包括label数量和分布等

├── raw_datasets            存放原始数据，分文件夹

├── total                   存放分组训练数据，直接服务于后续训练，从yolo_datasets中根据不同文件夹的分组情况进行组合，并切分train\valid\test子集

└── yolo_datasets           存放处理后的可用于yolo训练的数据，分文件夹，主要从raw_datasets中过滤非pair数据、labelme格式转换为coco格式等

本文件夹设计的思路是以数据集批次为单位，即分组是对现有数据集文件夹的划分
如果后续考虑对每个label类型分一个组，可新增一个total_label_oriented文件夹