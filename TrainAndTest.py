import yaml
import json
import os
from datetime import datetime  
from ultralytics import YOLOv10
import sys  
import cv2


from multiprocessing import Process, freeze_support

#（A）将labelme的标注数据格式修改为coco的格式
def xyxy2xywh(size, box):
    """
    convert [xmin, ymin, xmax, ymax] to [x_centre, y_centre, w, h]
    """
    w_all = size[0]
    h_all = size[1]

    top_left_x = box[0]
    top_left_y = box[1]
    bottom_right_x = box[2]
    bottom_right_y = box[3]
    

    w_tag = bottom_right_x - top_left_x
    h_tag = bottom_right_y - top_left_y
    x_center = top_left_x + w_tag / 2.0
    y_center = top_left_y + h_tag / 2.0


    w_tag_nom = w_tag / w_all
    h_tag_nom = h_tag / h_all
    x_center_nom = x_center / w_all
    y_center_nom = y_center / h_all

    return (x_center_nom, y_center_nom, w_tag_nom, h_tag_nom)

#（E）绘图
def draw( label_list, img, w1, h1, ouput_dir,  flag="five" ):
    # w1, h1来自图片img.shape[:2]
    used_list = []
    for label in label_list:
        if len(label)>0:
            if flag=="five":
                x, y, width, height = label[1:]
            else:
                x, y, width, height = label
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = xywh2xyxy(x, y, width, height, w1, h1)
            used_list.append( ( top_left_x, top_left_y, bottom_right_x, bottom_right_y ) )
    
    '''
    used_list = DataProcess.read_json("/home/jiangpeiwen2/jiangpeiwen2/workspace/CVDitect/datasets/封口环空洞/010.json")
    '''
    for label in used_list:
        top_left_x, top_left_y, bottom_right_x, bottom_right_y = label[0], label[1], label[2], label[3]
        # 绘图  rectangle()函数需要坐标为整数
        cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (0, 255, 0), 2)
    # cv2.imshow('show', img)
    cv2.imwrite(ouput_dir, img)
    cv2.waitKey(0)  # 按键结束
    cv2.destroyAllWindows()


def find_share_files_list(directory, type):  
    jpg_files = []  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if file.endswith( type ):  
                jpg_files.append(file[:-4])  
    return jpg_files

def get_root_path():
    if getattr(sys, 'frozen', False):  # 是否为PyInstaller打包的exe文件
        # 返回exe文件所在的绝对路径
        base_path = os.path.dirname(sys.executable)
    else:  # 在开发环境下运行
        # 返回脚本文件所在的绝对路径
        base_path = os.path.dirname( os.path.abspath(__file__)) 
    return base_path


class Model_Trainer():
    def __init__(self):
        self.RootPath = get_root_path()                # 项目根目录
        # self.RootPath = "", "home", "jiangpeiwen2", "jiangpeiwen2", "workspace", "ChipDetection"
        #（1）加载配置JSON文件
        config_path = os.path.join( self.RootPath, "config.json" )
        self.check_path_exsit( config_path, "请在项目根目录创建config.json配置文件！")
        with open( config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.groups = self.config["dataset_groups"]
        self.model_list = self.config["yolo10_model_all"]                          # 模型类型
        self.model_type = self.config["your_selected_model"]                       # 要训练的模型类型
        self.train_dataset_group = self.config["your_selected_group"]              # 要训练的数据集分组
        self.train_dataset_group_path = os.path.join( self.RootPath, "datasets", "total", f"{self.train_dataset_group}")  # 训练数据集所在路径
        #（2）加载统计JSON文件
        statistics_path = os.path.join( self.RootPath, f"datasets", "label_statistics.json")
        self.check_path_exsit( statistics_path, "datasets", "label_statistics.json配置文件不存在！")
        with open( statistics_path, "r", encoding="utf-8") as f:
            self.statistics = json.load(f)
        #（3）使用with语句打开（如果不存在则创建）一个yaml文件
        self.yaml_path = os.path.join( self.train_dataset_group_path, f"coco_chipdetection_{self.train_dataset_group}.yaml" )
        self.yaml_path_test = os.path.join( self.train_dataset_group_path, f"coco_chipdetection_{self.train_dataset_group}_test.yaml" )
        self.construct_yaml_file()
        #（4）检查是否有预训练模型权重并加载测试结果字典
        self.check_and_download_model()
        self.test_results_path = os.path.join( self.RootPath, "tests", "test_results.json")
        if os.path.exists( self.test_results_path ):
            with open( self.test_results_path, "r", encoding="utf-8") as f:
                self.test_results = json.load(f)
        else:
            self.test_results = {}
        
        
    def check_path_exsit( self, path, info, termination=True):
        if not os.path.exists( path ):
            if termination:
                raise Exception(f"{ info }！")
            else:
                print( f"{ info }！" )
                return False
        else:
            return True

    def construct_yaml_file( self ):
        target_dataset_label = self.statistics["groups_label"][self.train_dataset_group]
        train_config = {
            "path": self.train_dataset_group_path,
            "train": "images\\train",
            "val": "images\\valid",
            "test": "images\\test",
            # Classes
            "names": {},
            # 类别数量
            "nc": len(target_dataset_label)-1
        }
        _names = {}
        for key in target_dataset_label.keys():
            if key!="statistics":
                _names[ int(target_dataset_label[key])] = key
        train_config["names"] = _names
        with open( self.yaml_path, 'w') as file:  
            yaml.dump(train_config, file, allow_unicode=True)
        
        train_config["val"] = train_config["test"]
        with open( self.yaml_path_test, 'w') as file:  
            yaml.dump(train_config, file, allow_unicode=True)
    
    def check_and_download_model(self):
        #（1）检查模型权重文件
        self.yolo_type = self.config['your_yolo_number']
        self.pretrained_mode_path = os.path.join( self.RootPath, "models", "pre-trained", "Yolov10")
        if not self.check_path_exsit( self.pretrained_mode_path, f"{self.model_type}模型预训练权重不存在！", False):
            # 如果不存在整套模型权重，则调用下载脚本下载
            yolov10_download_shpath = os.path.join( self.RootPath, "models", "download_yolov10.sh" )
            status = os.system(f'bash {yolov10_download_shpath}') 
            if status == 0:  
                print("模型下载成功")  
            else:  
                print("模型下载失败")
        print( "模型权重文件检查完毕！")
    
    # 将所有测试集图片预测的方框都画上
    def show_predict(self, exp_name, model):
        test_data_path = os.path.join( self.train_dataset_group_path, f"images", "test")
        test_file_list = find_share_files_list( test_data_path, '.jpg')
        # 保存结果文件夹
        path_test_results = os.path.join( self.RootPath, "tests", "temp", f"{exp_name}")
        path_test_to_iamges = os.path.join(path_test_results, "test_images_tagged" )
        path_test_to_texts = os.path.join(path_test_results, "test_texts" )
        if not os.path.exists(path_test_results):
            os.makedirs( path_test_to_iamges )
            os.makedirs( path_test_to_texts )
        
        for file in test_file_list:
            from_path_file_image = os.path.join( test_data_path, f"{file}.jpg")
            img = cv2.imread( from_path_file_image )
            h, w = img.shape[:2]
            # 预测
            results = model( from_path_file_image )
            position_datas = []

            label_list = []
            for box in results[0].boxes:
                label = box.xyxy
                label = label.cpu()[0].tolist()
                yolo_label = xyxy2xywh( [w, h], label )
                label_list.append( yolo_label )
                line_str = ' '.join(map(str, label_list[0]))
                position_datas.append( f"{ int(box.cls)} {line_str}\n")
        
            with open( os.path.join(path_test_to_texts, f"{file}.txt"), "w", encoding="utf-8") as f:  
                for string in position_datas:  
                    # 将字符串写入文件，并在末尾添加换行符以确保每个字符串独占一行  
                    f.write(string)
            output_img_path = os.path.join(path_test_to_iamges, f"{file}.jpg")
            draw( label_list, img, w, h, output_img_path , flag="four")

    def train_test(self):
        model = YOLOv10( os.path.join( self.RootPath, "models", "pre-trained", "Yolov10", f"{self.model_type}.pt") )
        train_para = self.config["train_parameters"]
        # 训练结果保存的路径和文件夹名，同名会迭代地编号
        train_para["name"] = f"train_{self.model_type}_on_{self.train_dataset_group}_{train_para['epochs']}epoch_{train_para['batch']}batch"
        train_para["project"] = os.path.join( self.RootPath, f"models", "post-trained")
        train_para["device"] = "cpu"
        if not os.path.exists(train_para["project"]):
            os.makedirs(train_para["project"])
        print("**********************下面进行训练*********************************")
        
        results = model.train(data=self.yaml_path, **train_para)
        results = model.val( data=self.yaml_path )
        # Export the model to ONNX format
        success = model.export(format='onnx')
        # 测试
        print("**********************下面进行测试*********************************")
        metrics = model.val( data=self.yaml_path_test)
        now = datetime.now()  
        date_string = now.strftime( "%Y-%m-%d %H:%M:%S")
        result_name = f"{train_para['name']}_{date_string}"
        self.test_results[ result_name ] = metrics.results_dict
        with open( self.test_results_path, "w", encoding="utf-8") as f:
            json.dump( self.test_results, f, ensure_ascii=False, indent=4)
        """
        model = YOLOv10( "", "home", "jiangpeiwen2", "jiangpeiwen2", "workspace", "ChipDetection", "models", "post-trained", "train_yolov10n_on_封口环空洞_500epoch", "weights", "best.pt" )
        """
        # 展示测试结果，方便发现问题
        # test_result_path = os.path.join( train_para["project"], f"{train_para['name']}", "predictions.json")
        self.show_predict( train_para["name"], model )
        print("测试完毕")
            
def main( ):
    trainer = Model_Trainer()
    trainer.train_test()
  
if __name__ == "__main__":
    freeze_support()
    Process(target=main).start()
    
    