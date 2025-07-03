import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SpeakerDataset(Dataset):
    def __init__(self, processed_dir, speaker_folders, max_len=200):
        self.data = []
        self.labels = []
        self.label_map = {speaker: idx for idx, speaker in enumerate(speaker_folders)}

        for speaker in speaker_folders:
            speaker_path = os.path.join(processed_dir, speaker)
            for file in os.listdir(speaker_path):
                if file.endswith(".npy"):
                    mel_spec = np.load(os.path.join(speaker_path, file))

                    if mel_spec.shape[1] < max_len:
                        pad_width = max_len - mel_spec.shape[1]
                        mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode="constant")
                    elif mel_spec.shape[1] > max_len:
                        mel_spec = mel_spec[:, :max_len]

                    self.data.append(mel_spec)
                    self.labels.append(self.label_map[speaker])

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_spec = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_spec, label

def get_dataloader(processed_dir, speaker_folders, batch_size=32, max_len=200):
    dataset = SpeakerDataset(processed_dir, speaker_folders, max_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


