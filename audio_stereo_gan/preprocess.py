import os
import torch
import numpy as np
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """从立体声音频提取 M/S 信号的数据集"""

    def __init__(
        self,
        data_dirs,
        sr=44100,
        segment_length=8192,
        max_segments_per_file=4000,
    ):
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.data_dirs = data_dirs
        self.sr = sr
        self.segment_length = segment_length
        self.max_segments_per_file = max_segments_per_file

        self.files = []
        for data_dir in data_dirs:
            if not os.path.isdir(data_dir):
                continue
            for f in os.listdir(data_dir):
                if f.endswith(('.wav', '.flac', '.mp3')):
                    self.files.append(os.path.join(data_dir, f))

        self.segments = []
        self._prepare_segments()

    def _prepare_segments(self):
        """预处理所有音频文件，切分为固定长度片段"""
        for filepath in self.files:
            try:
                audio_np, sr = sf.read(filepath, always_2d=True)  # (samples, channels)
                audio = torch.from_numpy(audio_np.T).float()  # [channels, samples]
                target_sr = self.sr or sr
                if sr != target_sr:
                    audio = torchaudio.functional.resample(audio, sr, target_sr)
                if audio.shape[0] < 2:
                    continue  # 跳过单声道

                L, R = audio[0], audio[1]
                M = (L + R) / 2
                S = (L - R) / 2

                num_segments = M.numel() // self.segment_length
                if num_segments == 0:
                    continue

                max_segments = self.max_segments_per_file or num_segments
                if num_segments > max_segments:
                    # 均匀抽样，避免长音频导致训练过慢
                    indices = torch.linspace(0, num_segments - 1, steps=max_segments).long()
                else:
                    indices = torch.arange(num_segments)

                for i in indices:
                    start = i * self.segment_length
                    end = start + self.segment_length
                    self.segments.append((M[start:end], S[start:end]))
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        m, s = self.segments[idx]
        return m.float(), s.float()


def load_mono_audio(filepath, sr=44100, segment_length=8192):
    """加载单声道音频并返回 tensor"""
    audio_np, file_sr = sf.read(filepath, always_2d=True)
    audio = torch.from_numpy(audio_np.T).float()

    # 如果是立体声，先转成单声道
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)

    audio = audio.squeeze(0)
    if audio.numel() < segment_length:
        pad = segment_length - audio.numel()
        audio = torch.nn.functional.pad(audio, (0, pad))
    else:
        audio = audio[:segment_length]

    return audio.unsqueeze(0)


def load_stereo_audio(filepath, sr=44100):
    """加载立体声音频，返回 L, R 通道"""
    audio_np, file_sr = sf.read(filepath, always_2d=True)
    audio = torch.from_numpy(audio_np.T).float()

    if file_sr != sr:
        audio = torchaudio.functional.resample(audio, file_sr, sr)

    if audio.shape[0] == 1:
        # 如果输入是单声道，复制一份作为右声道，便于后续处理
        return audio[0], audio[0]

    return audio[0], audio[1]
