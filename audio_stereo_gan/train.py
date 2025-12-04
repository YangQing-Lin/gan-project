import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Generator, Discriminator
from preprocess import AudioDataset
from utils import plot_loss_curve


def train():
    # 设备选择：优先 MPS (Apple Silicon)，其次 CUDA，最后 CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 训练参数
    segment_length = 8192
    sr = 44100
    batch_size = 64
    epochs = 50
    lambda_l1 = 100.0

    # 加载数据集
    dataset = AudioDataset(
        ["data", "static"],
        sr=sr,
        segment_length=segment_length,
        max_segments_per_file=2000,
    )
    if len(dataset) == 0:
        print("No audio segments found. Please add stereo audio files to data/ or static/")
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print(f"Loaded {len(dataset)} audio segments")

    # 初始化模型
    gen = Generator(segment_length=segment_length).to(device)
    disc = Discriminator(segment_length=segment_length).to(device)

    # 优化器
    optim_gen = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optim_disc = torch.optim.Adam(disc.parameters(), lr=0.00005, betas=(0.5, 0.999))

    # 损失函数 - LSGAN (MSE)
    criterion = nn.MSELoss()

    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for m, s_real in pbar:
            m, s_real = m.to(device), s_real.to(device)
            batch_size = m.size(0)

            # === 训练判别器 ===
            fake_s = gen(m)

            # LSGAN 标签：real=1.0, fake=0.0
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # 判别真实 S
            real_output = disc(s_real)
            loss_real = criterion(real_output, real_labels)

            # 判别生成 S
            fake_output = disc(fake_s.detach())
            loss_fake = criterion(fake_output, fake_labels)

            loss_disc = (loss_real + loss_fake) / 2

            optim_disc.zero_grad()
            loss_disc.backward()
            optim_disc.step()

            # === 训练生成器（2次）===
            for _ in range(2):
                fake_s = gen(m)
                fake_output = disc(fake_s)
                loss_adv = criterion(fake_output, real_labels)
                loss_l1 = F.l1_loss(fake_s, s_real)
                loss_gen = loss_adv + lambda_l1 * loss_l1

                optim_gen.zero_grad()
                loss_gen.backward()
                torch.nn.utils.clip_grad_norm_(gen.parameters(), max_norm=5.0)
                optim_gen.step()

                epoch_g_loss += loss_gen.item()

            epoch_d_loss += loss_disc.item()

            pbar.set_postfix({
                'D_loss': f'{loss_disc.item():.4f}',
                'G_loss': f'{loss_gen.item():.4f}'
            })

        avg_g_loss = epoch_g_loss / (len(loader) * 2)  # G 训练 2 次
        avg_d_loss = epoch_d_loss / len(loader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        print(
            f"Epoch {epoch+1}/{epochs} - D_loss: {avg_d_loss:.4f}, "
            f"G_loss: {avg_g_loss:.4f}"
        )

        # 每轮更新损失曲线图
        plot_loss_curve(g_losses, d_losses)

    # 保存模型
    torch.save(gen.state_dict(), "gen.pth")
    torch.save(disc.state_dict(), "disc.pth")
    print("Models saved: gen.pth, disc.pth")

    # 保存 Loss 曲线
    plot_loss_curve(g_losses, d_losses)


if __name__ == "__main__":
    train()
