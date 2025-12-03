import argparse
import os
import torch

from model import Generator
from preprocess import load_mono_audio
from utils import save_stereo_audio, plot_waveform

OUTPUT_DIR = "output"


def inference(input_path, output_path, model_path="gen.pth"):
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 设备选择
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 加载生成器
    gen = Generator()
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.to(device)
    gen.train(False)  # 设置为评估模式
    print(f"Loaded model from {model_path}")

    # 加载单声道音频
    m = load_mono_audio(input_path).to(device)
    print(f"Loaded input audio: {input_path}")

    # 生成侧声道
    with torch.no_grad():
        s_pred = gen(m)

    # 合成立体声
    L = m + s_pred
    R = m - s_pred

    # 保存输出
    output_file = os.path.join(OUTPUT_DIR, os.path.basename(output_path))
    save_stereo_audio(L, R, output_file)

    # 可选：保存波形图
    plot_waveform(m, title="Mid Channel (M)", save_path=os.path.join(OUTPUT_DIR, "waveform_m.png"))
    plot_waveform(s_pred, title="Predicted Side Channel (S)", save_path=os.path.join(OUTPUT_DIR, "waveform_s.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate stereo from mono audio")
    parser.add_argument("--input", "-i", required=True, help="Input mono audio file")
    parser.add_argument("--output", "-o", default="output_stereo.wav", help="Output stereo file")
    parser.add_argument("--model", "-m", default="gen.pth", help="Generator model path")

    args = parser.parse_args()
    inference(args.input, args.output, args.model)
