import os
import random
random.seed(42)
import shutil
from tqdm import tqdm
import sys
import cv2
import numpy as np
import json

###################################################数据处理函数####################################################################
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

#（B）相反
def xywh2xyxy(x_center_nom, y_center_nom, w_tag_nom, h_tag_nom, w_all, h_all, need_norm=True):
    if need_norm:
        w_tag = w_tag_nom * w_all
        h_tag = h_tag_nom * h_all
        x_center = x_center_nom * w_all
        y_center = y_center_nom * h_all
    else:
        w_tag = w_tag_nom
        h_tag = h_tag_nom
        x_center = x_center_nom
        y_center = y_center_nom

    top_left_x = x_center - w_tag / 2.0
    top_left_y = y_center - h_tag / 2.0
    bottom_right_x = x_center + w_tag / 2.0
    bottom_right_y = y_center + h_tag / 2.0

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y

#（C）对yolo下的total进行图形标注，易于用户观察
def single_tag( from_path_file_image, num_labels, to_path_file_image_tagged, need_norm=True):
        # 读取image文件
        img = cv2.imdecode(np.fromfile( from_path_file_image, dtype=np.uint8), -1)
        h1, w1 = img.shape[:2]
        # 绘图并保存
        used_list = []
        for label in num_labels:
            if len(label)>0:
                x, y, width, height = label[1:]
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = xywh2xyxy(x, y, width, height, w1, h1, need_norm)
                used_list.append( ( top_left_x, top_left_y, bottom_right_x, bottom_right_y ) )
        for label in used_list:
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = label[0], label[1], label[2], label[3]
            # 绘图  rectangle()函数需要坐标为整数
            cv2.rectangle(img, (int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y)), (0, 255, 0), 2)
        # cv2.imshow('show', img)
        cv2.imwrite(to_path_file_image_tagged, img)
        cv2.waitKey(0)  # 按键结束
        cv2.destroyAllWindows()

#（D）返回前缀
def find_share_files_list(directory, type):  
    jpg_files = []  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if file.endswith( type ):  
                jpg_files.append(file[:-4])  
    return jpg_files

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
###################################################数据处理类####################################################################

def get_root_path():
    if getattr(sys, 'frozen', False):  # 是否为PyInstaller打包的exe文件
        # 返回exe文件所在的绝对路径
        base_path = os.path.dirname(sys.executable)
    else:  # 在开发环境下运行
        # 返回脚本文件所在的绝对路径
        base_path = os.path.dirname( os.path.abspath(__file__)) 
    return base_path



class DataProcess():
    def __init__(self):
        self.RootPath = get_root_path()               # 项目根目录
        #（1）加载配置config.json文件
        config_path = os.path.join( self.RootPath, "config.json" )
        if not os.path.exists( config_path ):
            raise Exception("请在项目根目录创建config.json配置文件！")
        with open( config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        self.groups = self.config["dataset_groups"]
        self.dataset_parts = ["train", "valid", "test"]
        self.dataset_parts_ratio = self.config["dataset_parts_ratio"]
        #（2）检查并创建三个主干子文件夹：如果不存在则创建：raw_datasets、yolo_datasets、total
        self.data_path = os.path.join( self.RootPath, f"datasets")                           # 项目数据目录
        self.raw_data_path = os.path.join( self.RootPath, f"datasets", "raw_datasets")          # raw数据目录，多个文件夹
        self.yolo_data_path = os.path.join( self.RootPath, f"datasets", "yolo_datasets")        # 对应yolo可用的数据目录，多个文件夹
        if not os.path.exists( self.yolo_data_path ):
            os.makedirs( self.yolo_data_path )
        self.yolo_data_path_total = os.path.join( self.data_path, f"total")                  # total文件夹
        if not os.path.exists( self.yolo_data_path_total ):
            os.makedirs( self.yolo_data_path_total )
        #（3）读取、创建数据集统计文件：三个固定键，其他为单个文件夹名为键
        label_statistics_path = os.path.join( self.data_path, "label_statistics.json")
        if not os.path.exists(label_statistics_path):       # 创建空文件
            self.dataset_label_map = {
                "total":{},
                "position_wrong": {},
                "groups_label": {}
            }
        else:                                               # 读取已有统计文件，含之前的结构信息
            with open( label_statistics_path, "r", encoding="utf-8") as f:
                self.dataset_label_map = json.load(f)
        #（4）确定本次变动涉及的数据组和子文件夹
        self.group_need_change = []
        self.file_need_change = []
        self.file_new = []

    #（1）读取新文件夹名称，并获取成对数据返回
    def read_file_name(self):
        #（1）读取raw_datasets和yolo_datasets数据集下子文件夹列表（每次的标签数据集单独放在一个子文件夹）
        raw_dirname_list = os.listdir( self.raw_data_path )
        yolo_dirname_list = os.listdir( self.yolo_data_path )
        ret_dict = {}
        #（2）遍历raw_datasets文件夹
        count = 0
        for i in range( len(raw_dirname_list) ):
            raw_name = raw_dirname_list[i]              # 在raw_datasets下的文件夹名
            yolo_name = raw_name+"_yolo"                # 在yolo_datasets下的文件夹名
            if os.path.isdir( os.path.join( self.raw_data_path, raw_name)) and yolo_name not in yolo_dirname_list:
                # 如果是文件夹，且其对应的_yolo名不在列表中，说明是新的数据集

                for group_n in self.groups.keys():
                    if raw_name in self.groups[group_n]:
                        self.group_need_change.append( group_n )
                        if group_n in self.group_need_change:
                            self.file_need_change.extend(self.groups[group_n])
                        break
                count +=1
                self.file_new.append(raw_name )
                print( f"发现新数据集{raw_name}")
                new_yolo_path = os.path.join( self.yolo_data_path, yolo_name)
                
                if not os.path.exists( new_yolo_path ):
                    # 为新的子文件夹创建在yolo_datasets文件夹下创建一个子文件夹
                    os.makedirs( os.path.join( new_yolo_path, "images"))
                    os.makedirs( os.path.join( new_yolo_path, "labels"))
                    os.makedirs( os.path.join( new_yolo_path, "images_tagged"))
        #（3）获取新数据集下所有json和jpg文件名列表，并检查是否有其他格式文件
        for raw_name in self.file_need_change:
            new_raw_path = os.path.join(self.raw_data_path, raw_name)
            raw_list_all = os.listdir( new_raw_path )
            jpg_list_prefix = []
            json_list_prefix = []
            for raw_file_name in raw_list_all:
                if raw_file_name.endswith( ".jpg" ):
                    jpg_list_prefix.append( raw_file_name[:-4])
                elif raw_file_name.endswith( ".json" ):
                    json_list_prefix.append( raw_file_name[:-5])
            if len(jpg_list_prefix) + len(json_list_prefix) != len(raw_list_all):
                raise Exception(f"{raw_name}数据集中有除json和jpg格式外的文件，请检查并修正。")
            #（4）获取该数据集中成对的数据：json和jpg格式兼备且文件名一致
            paired_file_name_prefix = []
            for prefix in json_list_prefix:
                if prefix in jpg_list_prefix:
                    paired_file_name_prefix.append( prefix )
            ret_dict[raw_name] = paired_file_name_prefix
        print(f"本次将处理{count}份数据集，文件夹名如下：")
        print( list(ret_dict.keys()) )
        return ret_dict

    #（2）处理单个json文件，返回coco格式坐标，并更新self.dataset_label_map
    def parse_json( self, data, raw_name, data_prefix ):
        if raw_name not in self.dataset_label_map["position_wrong"]:
            self.dataset_label_map["position_wrong"][raw_name] = []
        
        if raw_name not in self.dataset_label_map:
            self.dataset_label_map[raw_name] = {}

        label_dict = {}
        yolo_labels_num = []
        imageHeight = data["imageHeight"]
        imageWidth = data["imageWidth"]
        for shape in data["shapes"]:
            # 每个循环表示一个框
            normalized_points = [0, 0, 0, 0]                # [xmin, xmax, ymin, ymax]
            # 将json文件中的label数量加到self全局字典中，分为单个文件、全局、本地
            _label = shape["label"]
            # 仅新出现的数据集才进行增量统计
            if raw_name in self.file_new:                        
                if _label in self.dataset_label_map[raw_name]:
                    self.dataset_label_map[raw_name][_label] += 1
                else:
                    self.dataset_label_map[raw_name][_label] = 1
                
                if _label in self.dataset_label_map["total"]:
                    self.dataset_label_map["total"][_label] += 1
                else:
                    self.dataset_label_map["total"][_label] = 1
            if _label in label_dict:
                label_dict[_label] += 1
            else:
                label_dict[_label] = 1
            # 进行坐标转换
            index = 0
            for point in shape["points"]:
                normalized_points[0+index] = point[0]
                normalized_points[1+index] = point[1]
                index += 2
            if normalized_points[0]>normalized_points[2] or normalized_points[1]>normalized_points[3]:
                if data_prefix not in self.dataset_label_map["position_wrong"][raw_name]:
                    self.dataset_label_map["position_wrong"][raw_name].append( data_prefix )
                    print( f"坐标错误{raw_name} {data_prefix}")
        
                new_normalized_points = [ 0, 0, 0, 0]
                new_normalized_points[0] = min( normalized_points[0], normalized_points[2])
                new_normalized_points[2] = max( normalized_points[0], normalized_points[2])
                new_normalized_points[1] = min( normalized_points[1], normalized_points[3])
                new_normalized_points[3] = max( normalized_points[1], normalized_points[3])
                normalized_points = new_normalized_points
            yolo_label = xyxy2xywh( [imageWidth, imageHeight], normalized_points )      # yolo_labels_txt.append( f"{_label} {' '.join(map(str, yolo_label))}\n" )

            yolo_labels_num.append( [_label, yolo_label[0], yolo_label[1], yolo_label[2], yolo_label[3]])
        return yolo_labels_num, label_dict

    #（3）加载单个文件夹内的json格式文件并统计label数量
    def load_json_and_statistics(self, data_prefix, raw_name):
        json_dict = {}
        json_name = data_prefix+".json"
        json_path = os.path.join( self.raw_data_path,  f"{raw_name}", f"{json_name}")
        with open(json_path, 'r', encoding='utf-8') as json_file:  
            data = json.load(json_file)
        json_dict["position"], json_dict["label"]  = self.parse_json( data, raw_name, data_prefix)
        return json_dict

    #（4）全局，创建原始标签到index的分组映射
    def label_merge(self):
        notused_label = [ "r" ] 
        # 遍历每个需要改变的数据组
        for key in self.groups.keys():
            if key not in self.group_need_change:
                continue
            index = 0
            self.dataset_label_map["groups_label"][key] = {}
            # 遍历组中每个独立文件夹
            for _file in self.groups[key]:
                file_label = self.dataset_label_map[_file]
                # 遍历每个文件夹对应的label
                for label in file_label.keys():
                    # 如果这个label既不是被禁用的，之前也没被加入列表，则加入字典
                    if label not in notused_label and label not in self.dataset_label_map["groups_label"][key]:
                        self.dataset_label_map["groups_label"][key][label] = index
                        index +=1
        
    #（5）将原始label转变为分组的index label
    def replace_label(self, new_data_ready):
        ret_data_ready = {}
        for file_name in new_data_ready.keys():
            ret_data_ready[ file_name ] = {}
            group_name = None
            for group in self.groups.keys():
                if file_name in self.groups[group]:
                    group_name = group
                    break
            if group_name==None:
                raise Exception(f"没有找到{file_name}文件夹对应的组别")
            # 这是本组统一的新标签序号，用其替换所有
            label_dict = self.dataset_label_map["groups_label"][group_name]
            for prefix_name in new_data_ready[file_name]:
                ret_data_ready[ file_name ][prefix_name] = {
                    'position': [],
                    'label' : {}
                }
                all_labels_obj = new_data_ready[file_name][prefix_name]['position']
                for i in range( len(all_labels_obj) ):
                    original_position = new_data_ready[file_name][prefix_name]['position'][i]
                    if original_position[0] in label_dict:
                        original_position[0] = label_dict[ original_position[0] ]
                        ret_data_ready[file_name][prefix_name]['position'].append( original_position )
                        if original_position[0] not in ret_data_ready[file_name][prefix_name]['label']:
                            ret_data_ready[file_name][prefix_name]['label'][original_position[0]] = 1
                        else:
                            ret_data_ready[file_name][prefix_name]['label'][original_position[0]] += 1
                        
        return ret_data_ready

    #（6）将初步预处理完毕的数据pair保存到yolo系文件夹下（不划分train,test），以备后面训练使用
    def save_pair_yolo(self, index_new_data_ready ):
        for file_name in index_new_data_ready.keys():
            prefix_list = list( index_new_data_ready[file_name].keys() )
            for i in tqdm( range(len(prefix_list)), desc=f"Constructing Yolo Directory for {file_name}"):
                prefix = prefix_list[i]
                new_file_name = file_name + "_yolo"
                #（1）将image拷贝到images文件夹下
                image_from_path = os.path.join( self.raw_data_path, f"{file_name}", f"{prefix}.jpg")
                image_to_path = os.path.join( self.yolo_data_path, f"{new_file_name}", "images", f"{prefix}.jpg")
                shutil.copy(image_from_path, image_to_path) 
                #（2）将label列表以txt格式保存到labels文件夹下
                labels = index_new_data_ready[file_name][prefix]['position']
                label_to_path = os.path.join( self.yolo_data_path, f"{new_file_name}", "labels", f"{prefix}.txt")
                with open( label_to_path, "w", encoding="utf-8") as f:  
                    for label in labels:  
                        f.write( f"{' '.join(map(str, label))}\n")    # 将字符串写入文件，并在末尾添加换行符以确保每个字符串独占一行  
                #（3）对images进行标注放到images_tagged文件夹下
                tagged_image_to_path = os.path.join( self.yolo_data_path, f"{new_file_name}", "images_tagged", f"{prefix}.jpg")
                single_tag( image_from_path, labels, tagged_image_to_path)

    #（7-1）考虑多标签均衡的划分train_valid_test
    def split_multi_label(self, group_total):
        _train, _valid, _test = [], [], []

        index_vector = []
        dict_index = {}
        for i in range( len(group_total)):
            temp = [i]
            temp.extend( group_total[i][1])
            dict_index[i] = group_total[i][0]   # 建立Index和prefix的字典
            index_vector.append( temp )         # 使得index_vector称为一个int型的列表的列表
        """
        待完善
        """
        return _train, _valid, _test 
    
    #（7-2）简单划分train, valid, test
    def split_simple(self, file_list ):  
        random.shuffle( file_list )
        # ratio
        total_num = len( file_list )
        train_num = int( total_num*self.dataset_parts_ratio[0] )
        valid_num = int( total_num*self.dataset_parts_ratio[1] )
        # 切片
        _train = file_list[ :train_num ]
        _valid = file_list[ train_num: valid_num+train_num ]
        _test = file_list[ valid_num+train_num: ]
        return _train, _valid, _test

    #（8）将上述分文件夹处理好的文件，按照group整合到datasets/total目录下：仅对需要重新整理的进行整理
    def merge_yolo_for_train( self, index_new_data_ready ):
        # 遍历需要更改的group名
        for key in self.group_need_change:
            #（1）如果已经存在则要重新创建
            _dir_name = os.path.join( self.data_path, f"total", f"{key}")
            if os.path.exists( _dir_name ):
                try:  
                    shutil.rmtree(_dir_name)  
                    print(f"文件夹 {_dir_name} 及其内容已被成功删除。")  
                except OSError as e:  
                    print(f"删除文件夹时发生错误: {e.strerror}")
            os.makedirs( _dir_name )
            for _sub_dir in [ "images", "labels"]:
                for _sub_sub_dir in ["train", "test", "valid"]:
                    os.makedirs( os.path.join( _dir_name, f"{_sub_dir}", f"{_sub_sub_dir}") )
            #（2）将group所有子文件夹中的文件混合到一起，（混合文件名，label向量）
            sub_dirnames = self.groups[key]
            group_label = self.dataset_label_map["groups_label"][key]  # 本group所有标签
            group_total = []
            for sub_dirname in sub_dirnames:
                if sub_dirname not in self.groups[key]:
                    continue
                # 对该group中每个子文件夹进行处理
                sub_dir_dic = index_new_data_ready[sub_dirname]        # 单个文件夹数据，含每个文件的标签信息
                file_name_list = list( sub_dir_dic.keys() )            # 单个文件夹中所有文件名前缀
                for file_prefix in file_name_list:
                    # 对子文件夹中所有文件进行处理
                    single_label = [ 0 for i in range(len(group_label))]
                    mix_name = f"{sub_dirname}_yolo", f"{file_prefix}"          # 防止多个子文件夹有同名文件
                    _label_dic = sub_dir_dic[file_prefix]["label"]
                    for index_key in _label_dic:
                        single_label[index_key] = _label_dic[index_key]
                    group_total.append( (mix_name, single_label) )

            #（3）根据该group下所有子文件夹的基本情况进行split，返回值是group_total的同类数据结构
            _train, _valid, _test = self.split_simple( group_total )
            #（4）根据分割结果填充total目录
            _statistics = {
                "train":[ 0 for i in range(len(group_label))],
                "valid":[ 0 for i in range(len(group_label))],
                "test":[ 0 for i in range(len(group_label))],
            }
            for _part, _type in zip( (_train, _valid, _test), ["train", "valid", "test"]):
                for i in tqdm( range(len(_part)), desc=f"Moving for {_type} of group{key}"):
                    _prefix_name = _part[i][0]
                    _label_list = _part[i][1]
                    for j in range(len( group_label )):
                        _statistics[_type][j] += _label_list[j]
                    """
                    sub_dir = _prefix_name.split("/")[0]
                    prefix_name = _prefix_name.split("/")[1]
                    """
                    sub_dir = _prefix_name[0]
                    prefix_name = _prefix_name[1]
                    # 复制图像
                    from_image_path = os.path.join( self.yolo_data_path, f"{sub_dir}", "images", f"{prefix_name}.jpg")
                    to_image_path = os.path.join( self.data_path, f"total", f"{key}", f"images", f"{_type}", f"{prefix_name}.jpg")
                    shutil.copy(from_image_path, to_image_path) 
                    # 复制标签
                    from_label_path = os.path.join( self.yolo_data_path, f"{sub_dir}", "labels", f"{prefix_name}.txt")
                    to_label_path = os.path.join( self.data_path, f"total", f"{key}", "labels", f"{_type}", f"{prefix_name}.txt")
                    shutil.copy(from_label_path, to_label_path)
            self.dataset_label_map["groups_label"][key]["statistics"] = _statistics
            
    #（9）主函数
    def handler( self ):
        new_data_names = self.read_file_name()      # '引腿划伤标注'110, 引腿变色标注27, 封口环空洞标注101, X射线空洞标注74
        new_data_ready = {}
        # 遍历第一层，即多个文件夹
        for raw_file_name in new_data_names.keys():
            new_data_ready[raw_file_name] = {}
            filename_list = new_data_names[raw_file_name]
            # 遍历第二层，即每个文件夹内的prefix列表
            for file_name_prefix in filename_list:
                new_data_ready[raw_file_name][file_name_prefix] = self.load_json_and_statistics( file_name_prefix, raw_file_name )
        #（2）将原始label替换为分组标签，然后保存到本地，先不在数据集上分组
        self.label_merge()
        index_new_data_ready = self.replace_label( new_data_ready )
        self.save_pair_yolo(index_new_data_ready )
        #（3）将初步预处理的yolo系文件夹根据分组进行split
        self.merge_yolo_for_train( index_new_data_ready )
        #####　打印并保存　#####
        print(f"处理完毕，本次共处理{len(new_data_names.keys())}份数据集，标签情况如下")
        print( self.dataset_label_map )
        with open( os.path.join( self.data_path, "label_statistics.json"), "w", encoding="utf-8") as f:
            json.dump( self.dataset_label_map, f, ensure_ascii=False, indent=4)
        #return index_new_data_ready, new_data_names


if __name__ == "__main__":
    DataHandle = DataProcess()
    DataHandle.handler()
    
    
   