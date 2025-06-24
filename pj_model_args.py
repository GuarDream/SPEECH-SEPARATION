import torch
import yaml
import argparse
from look2hear.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

def print_model_args(model_path):
    try:
        # 加载模型配置文件
        model_conf = torch.load(model_path, map_location="cpu")
        # 提取 model_args 参数
        model_args = model_conf["model_args"]
        print("Model arguments:")
        print(model_args)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description="Load model and print model arguments.")
parser.add_argument("--model_path", default="/home/pj/Desktop/TIGER-main/Experiments/checkpoint/TIGER-EchoSet/best_model.pth", help="Path to the pre-trained model.")
# 获取模型保存路径
args = parser.parse_args()

model_path = args.model_path
# 打印 model_args 参数
print_model_args(model_path)

