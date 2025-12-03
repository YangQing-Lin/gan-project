import argparse
import os
import numpy as np
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from model import Generator

OUTPUT_DIR = "output"


def validate(input_stereo, model_path="gen.pth", sr=44100):
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

    # 加载原始立体声
    audio_np, file_sr = sf.read(input_stereo, always_2d=True)  # (samples, channels)
    audio = torch.from_numpy(audio_np.T).float()  # [channels, time]

    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)

    if audio.shape[0] < 2:
        print("Error: 输入必须是双声道音频")
        return

    L, R = audio[0], audio[1]
    M_real = ((L + R) / 2).numpy()
    S_real = ((L - R) / 2).numpy()

    # 加载模型
    gen = Generator()
    gen.load_state_dict(torch.load(model_path, map_location=device))
    gen.to(device)
    gen.train(False)

    # 分段预测（每段 16384 采样点）
    segment_length = 8192
    S_pred_list = []

    for i in range(0, len(M_real), segment_length):
        m_seg = M_real[i:i+segment_length]
        if len(m_seg) < segment_length:
            m_seg = np.pad(m_seg, (0, segment_length - len(m_seg)))

        m_tensor = torch.tensor(m_seg, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            s_pred = gen(m_tensor).cpu().numpy().squeeze()
        S_pred_list.append(s_pred[:min(segment_length, len(M_real) - i)])

    S_pred = np.concatenate(S_pred_list)[:len(S_real)]

    # 计算指标
    mse = np.mean((S_real - S_pred) ** 2)
    correlation = np.corrcoef(S_real, S_pred)[0, 1]

    print(f"\n=== 验证结果 ===")
    print(f"MSE (均方误差): {mse:.6f}  (越小越好)")
    print(f"相关系数: {correlation:.4f}  (越接近 1 越好)")

    # 绘制对比图
    fig, axes = plt.subplots(3, 1, figsize=(14, 8))
    time = np.arange(len(S_real)) / sr

    axes[0].plot(time, S_real, linewidth=0.5)
    axes[0].set_title("Original Side Channel (S_real)")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(time, S_pred, linewidth=0.5, color='orange')
    axes[1].set_title("Predicted Side Channel (S_pred)")
    axes[1].set_ylabel("Amplitude")

    axes[2].plot(time, S_real - S_pred, linewidth=0.5, color='red')
    axes[2].set_title("Difference (S_real - S_pred)")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Amplitude")

    plt.tight_layout()
    result_path = os.path.join(OUTPUT_DIR, "validation_result.png")
    plt.savefig(result_path)
    print(f"\n对比图已保存: {result_path}")

    # 评价
    print(f"\n=== 评价 ===")
    if correlation > 0.7:
        print("✅ 优秀：预测与真实高度相关")
    elif correlation > 0.4:
        print("⚠️ 一般：有一定相关性，可继续训练")
    else:
        print("❌ 较差：预测与真实基本不相关，需调整模型或数据")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证模型效果")
    parser.add_argument("--input", "-i", required=True, help="原始双声道音频文件")
    parser.add_argument("--model", "-m", default="gen.pth", help="模型路径")

    args = parser.parse_args()
    validate(args.input, args.model)
