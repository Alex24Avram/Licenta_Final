import os
import pickle
from torch.utils.data import random_split, DataLoader
from preprocess import preprocess_audio
from feature_extraction import get_dataloader

def prepare_datasets(dataset_path, output_dir, speaker_folders, num_files_to_combine, batch_size=32):
    os.makedirs(output_dir, exist_ok=True)

    preprocess_audio(dataset_path, output_dir, speaker_folders, num_files_to_combine, n_mels=128, sr=16000)
    print("Preprocesare finalizata!")

    full_dataset = get_dataloader(output_dir, speaker_folders, batch_size=batch_size).dataset

    train_size = int(0.7 * len(full_dataset))
    test_size = int(0.15 * len(full_dataset))
    val_size = len(full_dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = random_split(full_dataset, [train_size, test_size, val_size])

    with open(os.path.join(output_dir, "dataset_splits.pkl"), "wb") as f:
        pickle.dump((train_dataset, test_dataset, val_dataset), f)
    
    print("Seturile de date au fost salvate.")

if __name__ == "__main__":
    dataset_path = "E:\\Anul 4\\Dataset\\16000_pcm_speeches"
    output_dir = "E:\\Anul 4\\Dataset\\work"
    speaker_folders = [
        "Benjamin_Netanyau",
        "Jens_Stoltenberg",
        "Julia_Gillard",
        "Magaret_Tarcher",
        "Nelson_Mandela"
    ]
    num_files_to_combine = 1500
    prepare_datasets(dataset_path, output_dir, speaker_folders, num_files_to_combine)
    print("Proces finalizat!")
