# 获取脚本的完整路径  
SCRIPT_PATH=$(realpath "$0")  
# 获取脚本父目录的父目录  
ROOT=$(dirname "$(dirname "$SCRIPT_PATH")")
download_path="$ROOT/models/pre-trained"

cd $download_path
git lfs install
git clone https://www.modelscope.cn/THU-MIG/Yolov10.git