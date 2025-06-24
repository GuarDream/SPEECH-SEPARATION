import yaml
import os
import look2hear.models
import argparse
import torch
import torchaudio
import debugpy
import datetime

# Debugging
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# audio path
parser = argparse.ArgumentParser()
parser.add_argument("--audio_path", default="test/test_mixture_466.wav", help="Path to audio file")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# # Load model
try:
    model = look2hear.models.TIGERDNR.from_pretrained("JusperLee/TIGER-DnR", cache_dir="cache", local_files_only=True)
    debugpy.breakpoint()  # 手动中断后检查变量状态
    print("模型状态:", model.state_dict().keys())  # 检查是否加载成功
    print("Loading TIGER-DnR model...")
except Exception as e:
    debugpy.breakpoint()
    print("Failed to load TIGER-DnR model.")
    print("Error loading model, please check the model path or internet connection.")
    print(e)
    exit(1)
if not os.path.exists("cache"):
    os.makedirs("cache")
print("Cache directory created at 'cache'.")

try:
    model.to(device)
    print("Model transferred to device:", device)
except Exception as e:  # Handle device transfer errors
    debugpy.breakpoint()
    print("Failed to transfer model to device.")
    print("Error transferring model, please check the device compatibility.")
    print(e)
    exit(1)
try:
    model.eval()
    print("Model set to evaluation mode.")
except Exception as e:
    debugpy.breakpoint()
    print("Failed to set model to evaluation mode.")
    print("Error setting model to eval mode, please check the model configuration.")
    print(e)
    exit(1)
try:
    audio = torchaudio.load(parser.parse_args().audio_path)[0].to(device)
    print("Audio loaded successfully.")
except Exception as e:
    debugpy.breakpoint()
    print("Failed to load audio file.")
    print("Error loading audio file, please check the file path or format.")
    print(e)
    exit(1)
try:
    with torch.no_grad():
        all_target_dialog, all_target_effect, all_target_music = model(audio[None])
        print("Audio processed successfully.")
except Exception as e:
    debugpy.breakpoint()
    print("Failed to process audio with the model.")
    print("Error processing audio, please check the model input format or audio data.")
    print(e)
    exit(1)

# Save the separated audio files
try:
   # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存文件
    torchaudio.save(f"test/test_target_dialog_466_{timestamp}.wav", all_target_dialog.cpu(), 48000)
    torchaudio.save(f"test/test_target_effect_466_{timestamp}.wav", all_target_effect.cpu(), 48000)
    torchaudio.save(f"test/test_target_music_466_{timestamp}.wav", all_target_music.cpu(), 48000)
    print("Separated audio files saved successfully.")
    print("GPU显存占用:", torch.cuda.memory_allocated(device) / 1024**2, "MB")
except Exception as e:
    debugpy.breakpoint()
    print("Failed to save separated audio files.")
    print("Error saving audio files, please check the output directory or file permissions.")
    print(e)
    exit(1)
# Print GPU memory usage
