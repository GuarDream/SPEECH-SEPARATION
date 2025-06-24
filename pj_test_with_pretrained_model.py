import os
import argparse
import torch
import torchaudio
import yaml
import look2hear.models

# 解析命令行参数
parser = argparse.ArgumentParser(description="Load model and save output audio.")
parser.add_argument("--audio_path", default="test/mix.wav", help="Path to audio file (mixture).")
parser.add_argument("--output_dir", default="separated_audio", help="Directory to save separated audio files.")
parser.add_argument("--model_path", default="Experiments/checkpoint/TIGER-EchoSet/best_model.pth", help="Path to the pre-trained model.")
parser.add_argument("--conf_dir", default="/home/pj/Desktop/TIGER-main/configs/tiger.yml", help="Path to the configuration file.")
args = parser.parse_args()

# 加载配置文件
with open(args.conf_dir, "rb") as f:
    config = yaml.safe_load(f)

# 设备设置
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# 加载模型
print("Loading TIGER model...")
model = getattr(look2hear.models, config["audionet"]["audionet_name"]).from_pretrain(
    args.model_path,
    sample_rate=config["datamodule"]["data_config"]["sample_rate"],
    **config["audionet"]["audionet_config"],
)
model.to(device)
model.eval()

# 音频加载和预处理
target_sr = config["datamodule"]["data_config"]["sample_rate"]
print(f"Loading audio from: {args.audio_path}")
try:
    waveform, original_sr = torchaudio.load(args.audio_path)
except Exception as e:
    print(f"Error loading audio file {args.audio_path}: {e}")
    exit(1)
print(f"Original sample rate: {original_sr} Hz, Target sample rate: {target_sr} Hz")

# 重采样
if original_sr != target_sr:
    print(f"Resampling audio from {original_sr} Hz to {target_sr} Hz...")
    resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
    waveform = resampler(waveform)
    print("Resampling complete.")

# 将音频移动到目标设备
audio = waveform.to(device)

# 准备输入张量
if audio.dim() == 1:
    audio = audio.unsqueeze(0)  # 添加通道维度 -> [1, T]
audio_input = audio.unsqueeze(0).to(device)
print(f"Audio tensor prepared with shape: {audio_input.shape}")

# 语音分离
print("Performing separation...")
with torch.no_grad():
    ests_speech = model(audio_input)  # 预期输出形状: [B, num_spk, T]

# 处理分离后的语音
ests_speech = ests_speech.squeeze(0)
num_speakers = ests_speech.shape[0]
print(f"Separation complete. Detected {num_speakers} potential speakers.")

# 创建输出目录
os.makedirs(args.output_dir, exist_ok=True)
print(f"Output directory: {args.output_dir}")

# 保存分离后的音频文件
for i in range(num_speakers):
    output_filename = os.path.join(args.output_dir, f"channel_ti_{i + 1}.wav")
    speaker_track = ests_speech[i].cpu().unsqueeze(0)
    print(f"Saving speaker {i + 1} to {output_filename}")
    try:
        torchaudio.save(output_filename, speaker_track, target_sr)
    except Exception as e:
        print(f"Error saving file {output_filename}: {e}")