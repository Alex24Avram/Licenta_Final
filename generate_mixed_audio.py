import os
import random
import math
import librosa
import soundfile as sf
import json
import numpy as np

DATASET_PATH = "E://Anul 4//Dataset//16000_pcm_speeches"
OUTPUT_DIR = "E://Anul 4//Dataset//mixed"
SAMPLE_RATE = 16000

def list_wav_files(speaker_dir: str) -> list[str]:
    wavs = [f for f in os.listdir(speaker_dir) if f.lower().endswith(".wav")]
    if not wavs:
        raise FileNotFoundError(f"Nu exista fisiere .wav in \n  {speaker_dir}")
    try:
        wavs_sorted = sorted(wavs, key=lambda x: int(os.path.splitext(x)[0]))
    except ValueError:
        wavs_sorted = sorted(wavs)
    return wavs_sorted

def generate_session_audio(
    output_audio_path: str,
    output_label_path: str,
    schedule: list[tuple[str, float]],
    sr: int = SAMPLE_RATE
):
    os.makedirs(os.path.dirname(output_audio_path), exist_ok=True)

    speaker_files: dict[str, list[str]] = {}
    for (sp, _) in schedule:
        if sp in speaker_files:
            continue
        speaker_dir = os.path.join(DATASET_PATH, sp)
        if not os.path.isdir(speaker_dir):
            raise FileNotFoundError(f"Folderul pentru speaker '{sp}' nu exista:\n  {speaker_dir}")
        wav_files = list_wav_files(speaker_dir)
        speaker_files[sp] = wav_files

    final_audio_segments: list[np.ndarray] = []
    timeline: list[dict] = []
    current_time = 0.0

    for (sp, turn_dur) in schedule:
        wav_list = speaker_files[sp]
        total_files = len(wav_list)
        needed_samples = int(turn_dur * sr)
        samples_per_file = sr
        needed_files = math.ceil(needed_samples / samples_per_file)

        if needed_files <= total_files:
            max_start_file = total_files - needed_files
            start_file_idx = random.randint(0, max_start_file)
            candidate_files = wav_list[start_file_idx : start_file_idx + needed_files]
        else:
            repeats = math.ceil(needed_files / total_files)
            extended_list = wav_list * repeats
            max_start_file = len(extended_list) - needed_files
            start_file_idx = random.randint(0, max_start_file)
            candidate_files = extended_list[start_file_idx : start_file_idx + needed_files]

        seg_buffer = np.array([], dtype=np.float32)
        speaker_dir = os.path.join(DATASET_PATH, sp)
        for fname in candidate_files:
            path_wav = os.path.join(speaker_dir, fname)
            y, _ = librosa.load(path_wav, sr=sr)
            seg_buffer = np.concatenate([seg_buffer, y.astype(np.float32)])

        segment = seg_buffer[:needed_samples]
        final_audio_segments.append(segment)

        timeline.append({
            "speaker": sp,
            "start": round(current_time, 2),
            "end": round(current_time + turn_dur, 2)
        })
        current_time += turn_dur

    mixed = np.concatenate(final_audio_segments, axis=0)
    sf.write(output_audio_path, mixed, sr)

    with open(output_label_path, 'w', encoding='utf-8') as f:
        json.dump(timeline, f, indent=4, ensure_ascii=False)

    print(f"Inregistrare generata: {output_audio_path}")
    print(f"Etichete salvate:     {output_label_path}")

#"Benjamin_Netanyau"
#"Julia_Gillard"
#"Nelson_Mandela"
#"Jens_Stoltenberg"
#"Magaret_Tarcher"


if __name__ == '__main__':
    schedule = [
    ("Benjamin_Netanyau", 14.0),
    ("Julia_Gillard", 12.0),
    ("Benjamin_Netanyau", 10.0),
    ("Nelson_Mandela", 13.0),
    ("Benjamin_Netanyau", 15.0),
    ("Magaret_Tarcher", 14.0),
    ("Benjamin_Netanyau", 14.0),
    ("Magaret_Tarcher", 8.0),
    ("Nelson_Mandela", 11.0),
    ("Jens_Stoltenberg", 8.0),
    ("Julia_Gillard", 10.0),
    ("Benjamin_Netanyau", 10.0),
    ("Jens_Stoltenberg", 14.0),
    ("Julia_Gillard", 8.0),
    ("Magaret_Tarcher", 9.0),
    ("Benjamin_Netanyau", 11.0),
    ("Magaret_Tarcher", 10.0),
    ("Jens_Stoltenberg", 8.0),
    ("Magaret_Tarcher", 11.0),
    ("Benjamin_Netanyau", 9.0),
    ("Julia_Gillard", 12.0),
    ("Magaret_Tarcher", 10.0),
    ("Benjamin_Netanyau", 11.0),
    ("Magaret_Tarcher", 8.0),
]
 
    filename_base = "5.30"
    out_wav  = os.path.join(OUTPUT_DIR, f"{filename_base}.wav")
    out_json = os.path.join(OUTPUT_DIR, f"{filename_base}_labels.json")

    generate_session_audio(out_wav, out_json, schedule)




