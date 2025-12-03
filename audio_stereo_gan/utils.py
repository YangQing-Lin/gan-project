import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


def save_stereo_audio(L, R, filepath, sr=22050):
    """保存立体声音频文件"""
    if hasattr(L, 'cpu'):
        L = L.cpu().detach().numpy()
    if hasattr(R, 'cpu'):
        R = R.cpu().detach().numpy()

    L = np.squeeze(L)
    R = np.squeeze(R)

    stereo = np.stack([L, R], axis=-1)
    sf.write(filepath, stereo, sr)
    print(f"Saved stereo audio to {filepath}")


def plot_loss_curve(g_losses, d_losses, save_path="loss_curve.png"):
    """绘制并保存 Loss 曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Adv Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def plot_waveform(audio, sr=22050, title="Waveform", save_path=None):
    """绘制音频波形"""
    if hasattr(audio, 'cpu'):
        audio = audio.cpu().detach().numpy()
    audio = np.squeeze(audio)

    plt.figure(figsize=(12, 4))
    time = np.arange(len(audio)) / sr
    plt.plot(time, audio)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        print(f"Waveform saved to {save_path}")
    plt.close()
