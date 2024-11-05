import tkinter as tk  
from tkinter import messagebox  
import subprocess  
import os 
import sys

def get_root_path():
    if getattr(sys, 'frozen', False):  # 是否为PyInstaller打包的exe文件
        # 返回exe文件所在的绝对路径
        base_path = os.path.dirname(sys.executable)
    else:  # 在开发环境下运行
        # 返回脚本文件所在的绝对路径
        base_path = os.path.dirname( os.path.abspath(__file__)) 
    return base_path


  
class App:  
    def __init__(self, root):  
        self.root = root  
        self.root.title("可执行文件运行器")  
        self.root.geometry("300x200")  
  
        # 获取当前脚本所在的目录  
        self.current_directory = get_root_path() 
  
        # 创建按钮  
        self.data_processor_button = tk.Button(root, text="运行 DataProcessor.exe", command=self.run_data_processor)  
        self.data_processor_button.pack(pady=20)  
  
        self.train_and_test_button = tk.Button(root, text="运行 TrainAndTest.exe", command=self.run_train_and_test)  
        self.train_and_test_button.pack(pady=20)  
  
    def run_data_processor(self):  
        exe_path = os.path.join(self.current_directory, "DataProcessor.exe")  
        self.run_exe(exe_path)  
  
    def run_train_and_test(self):  
        exe_path = os.path.join(self.current_directory, "TrainAndTest.exe")  
        self.run_exe(exe_path)  
  
    def run_exe(self, exe_path):  
        try:  
            # 使用 subprocess.Popen 在 CMD 窗口中运行 EXE 文件  
            subprocess.Popen([exe_path], shell=True)  
            messagebox.showinfo("信息", f"{os.path.basename(exe_path)} 已启动！")  
        except FileNotFoundError:  
            messagebox.showerror("错误", f"{os.path.basename(exe_path)} 未找到，请检查文件路径。")  
        except Exception as e:  
            messagebox.showerror("错误", f"运行时发生错误: {str(e)}")  
  
if __name__ == "__main__":  
    root = tk.Tk()  
    app = App(root)  
    root.mainloop()