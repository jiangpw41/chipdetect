from ultralytics import YOLOv10
import os

_ROOT_PATH = os.path.dirname( os.path.dirname(__file__) )
# Create a new YOLO model from scratch


pretrained_model_path = "/home/jiangpeiwen2/jiangpeiwen2/workspace/ChipDetection/models/pre-trained/Yolov10/yolov10n.pt"
coco_yaml_path = "/home/jiangpeiwen2/jiangpeiwen2/workspace/YOLO/datasets/coco_SealingRingCavity.yaml"

model = YOLOv10( pretrained_model_path )

results = model.train(data=coco_yaml_path, epochs=500, batch=512, imgsz=640, device='0,1,2,3,4,5,6,7')
 
# Evaluate the model's performance on the validation set
results = model.val()

# Export the model to ONNX format
success = model.export(format='onnx')