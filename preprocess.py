import os
import librosa
import numpy as np
import noisereduce as nr

def preprocess_audio(dataset_path, output_path, speaker_folders, num_files_to_combine, n_mels=128, sr=16000):
    os.makedirs(output_path, exist_ok=True)

    for speaker in speaker_folders:
        speaker_path = os.path.join(dataset_path, speaker)
        output_speaker_dir = os.path.join(output_path, speaker)
        os.makedirs(output_speaker_dir, exist_ok=True)

        wav_files = [f"{i}.wav" for i in range(num_files_to_combine)]
        for wav_file in wav_files:
            wav_file_path = os.path.join(speaker_path, wav_file)
            if not os.path.exists(wav_file_path):
                print(f"File not found: {wav_file_path}")
                continue

            audio, sr = librosa.load(wav_file_path, sr=sr)
            
            # Eliminare zgomot
            audio = nr.reduce_noise(y=audio, sr=sr, n_fft=512, win_length=256, n_jobs=-1)

            # Eliminare segmente fara vorbire
            intervals = librosa.effects.split(audio, top_db=30)
            non_silent_audio = np.concatenate([audio[start:end] for start, end in intervals])

            if len(non_silent_audio) < 1:
                print(f"No speech detected in {wav_file_path}")
                continue

            # Convertire la mel-spectrograma
            mel_spec = librosa.feature.melspectrogram(y=non_silent_audio, sr=sr, n_mels=n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            # Salvare spectrograma ca .npy
            npy_filename = os.path.join(output_speaker_dir, f"{wav_file.replace('.wav', '.npy')}")
            np.save(npy_filename, mel_spec_db)
            print(f"Saved Mel-Spectrogram: {npy_filename}")



