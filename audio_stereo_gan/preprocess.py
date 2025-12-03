import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    """从立体声音频提取 M/S 信号的数据集"""

    def __init__(self, data_dir, sr=22050, segment_length=16384):
        self.data_dir = data_dir
        self.sr = sr
        self.segment_length = segment_length
        self.files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(('.wav', '.flac', '.mp3'))
        ]
        self.segments = []
        self._prepare_segments()

    def _prepare_segments(self):
        """预处理所有音频文件，切分为固定长度片段"""
        for filepath in self.files:
            try:
                audio, _ = librosa.load(filepath, sr=self.sr, mono=False)
                if audio.ndim == 1:
                    continue  # 跳过单声道
                L, R = audio[0], audio[1]
                M = (L + R) / 2
                S = (L - R) / 2

                num_segments = len(M) // self.segment_length
                for i in range(num_segments):
                    start = i * self.segment_length
                    end = start + self.segment_length
                    self.segments.append((M[start:end], S[start:end]))
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        m, s = self.segments[idx]
        return (
            torch.tensor(m, dtype=torch.float32),
            torch.tensor(s, dtype=torch.float32)
        )


def load_mono_audio(filepath, sr=22050, segment_length=16384):
    """加载单声道音频并返回 tensor"""
    audio, _ = librosa.load(filepath, sr=sr, mono=True)
    if len(audio) < segment_length:
        audio = np.pad(audio, (0, segment_length - len(audio)))
    else:
        audio = audio[:segment_length]
    return torch.tensor(audio, dtype=torch.float32).unsqueeze(0)


def load_stereo_audio(filepath, sr=22050):
    """加载立体声音频，返回 L, R 通道"""
    audio, _ = librosa.load(filepath, sr=sr, mono=False)
    if audio.ndim == 1:
        return audio, audio
    return audio[0], audio[1]
