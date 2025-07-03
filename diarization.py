import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from sklearn.metrics import adjusted_rand_score, confusion_matrix, ConfusionMatrixDisplay

from model import get_model
from clustering_methods import diarize_kmeans, diarize_spectral, estimate_clusters

AUDIO_DIR = "E:/Anul 4/Dataset/mixed"
RESULTS_DIR = os.path.join(AUDIO_DIR, "Results", "Update")
os.makedirs(RESULTS_DIR, exist_ok=True)

METHODS = ["kmeans", "spectral"]
MODEL_TYPES = ["simple", "baseline", "resnet"]
SEGMENT_DURATION = 1.0
SAMPLE_RATE = 16000
NUM_SPEAKERS = 5

def segment_audio(audio, sr, segment_duration):
    samples = int(segment_duration * sr)
    return [audio[i*samples:(i+1)*samples] for i in range(len(audio)//samples)]

@torch.no_grad()
def extract_embeddings(segments, model, sr, max_len=200):
    model.eval()
    device = next(model.parameters()).device
    embs = []
    for seg in segments:
        mel = librosa.feature.melspectrogram(y=seg, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] < max_len:
            pad = max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0,0),(0,pad)), mode="constant")
        else:
            mel_db = mel_db[:, :max_len]
        inp = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        emb = model.extract_embedding(inp).squeeze()
        if emb.ndim > 1:
            emb = emb.view(emb.size(0), -1)[0]
        embs.append(emb.cpu().numpy())
    return np.array(embs)

def compute_der(gt, pred):
    valid = [i for i,x in enumerate(gt) if x != -1]
    return sum(gt[i]!=pred[i] for i in valid)/len(valid) if valid else 1.0

def load_ground_truth(path, total_segments, segment_duration):
    data = json.load(open(path))
    gt = [-1]*total_segments
    sp2id = {}
    idx = 0
    for e in data:
        sp = e['speaker']
        if sp not in sp2id:
            sp2id[sp] = idx; idx+=1
        start = int(e['start']//segment_duration)
        end = int(e['end']//segment_duration)
        for i in range(start, min(end, total_segments)):
            gt[i] = sp2id[sp]
    return np.array(gt)

def plot_diarization(gt, pred, segment_duration, out_path):
    fig, axs = plt.subplots(2,1,sharex=True, figsize=(12,4))
    x = np.arange(len(gt))*segment_duration
    axs[0].bar(x, [1]*len(gt), width=segment_duration, color=plt.cm.tab10(gt), align='edge')
    axs[0].set_title('Ground Truth'); axs[0].set_yticks([])
    axs[1].bar(x, [1]*len(pred), width=segment_duration, color=plt.cm.tab10(pred), align='edge')
    axs[1].set_title('Predicted'); axs[1].set_yticks([])
    plt.xlabel('Time (s)'); plt.tight_layout(); plt.savefig(out_path); plt.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics = []
    method_metrics = {m: [] for m in METHODS}

    for METHOD in METHODS:
        for MODEL_TYPE in MODEL_TYPES:
            model_dir = os.path.join(RESULTS_DIR, METHOD, MODEL_TYPE)
            os.makedirs(model_dir, exist_ok=True)
            model = get_model(MODEL_TYPE, NUM_SPEAKERS).to(device)
            model.load_state_dict(torch.load(f"trained_model_{MODEL_TYPE}.pth", map_location=device))

            for fname in sorted(os.listdir(AUDIO_DIR)):
                if not fname.endswith('.wav'):
                    continue
                print(f"\nSe proceseaza {fname} [{METHOD}/{MODEL_TYPE}]")
                y, sr = librosa.load(os.path.join(AUDIO_DIR, fname), sr=SAMPLE_RATE)
                segs = segment_audio(y, sr, SEGMENT_DURATION)
                embs = extract_embeddings(segs, model, sr)
                gt = load_ground_truth(os.path.join(AUDIO_DIR, fname.replace('.wav','_labels.json')), len(embs), SEGMENT_DURATION)

                if METHOD == 'kmeans':
                    pred = diarize_kmeans(embs, len(set(gt[gt != -1])), gt)
                else:
                    pred = diarize_spectral(embs, gt)

                ari = adjusted_rand_score(gt, pred)
                der = compute_der(gt, pred)
                print(f"ARI: {ari:.4f}, DER: {der:.4f}")

                all_metrics.append({
                    'method': METHOD,
                    'model': MODEL_TYPE,
                    'file': fname,
                    'ARI': round(ari, 4),
                    'DER': round(der, 4)
                })
                method_metrics[METHOD].append((ari, der))

                diar = {}
                cur, start = pred[0], 0.0
                for i in range(1, len(pred)):
                    if pred[i] != cur:
                        diar.setdefault(f"Speaker {cur}", []).append({'start': round(start, 2), 'end': round(i * SEGMENT_DURATION, 2)})
                        cur = pred[i]; start = i * SEGMENT_DURATION
                diar.setdefault(f"Speaker {cur}", []).append({'start': round(start, 2), 'end': round(len(pred) * SEGMENT_DURATION, 2)})
                with open(os.path.join(model_dir, fname.replace('.wav', '_diarization.json')), 'w') as jf:
                    json.dump(diar, jf, indent=4)
                out_plot = os.path.join(model_dir, fname.replace('.wav', '_diarization.png'))
                plot_diarization(gt, pred, SEGMENT_DURATION, out_plot)
                cm_idx = [i for i, x in enumerate(gt) if x != -1]
                cm = confusion_matrix([gt[i] for i in cm_idx], [pred[i] for i in cm_idx])
                disp = ConfusionMatrixDisplay(cm)
                disp.plot(cmap='Blues')
                cmp_path = os.path.join(model_dir, fname.replace('.wav', '_confusion.png'))
                plt.tight_layout(); 
                plt.savefig(cmp_path); 
                plt.close()

    metrics_file = os.path.join(RESULTS_DIR, 'metrics_per_recording.json')
    with open(metrics_file, 'w') as mf:
        json.dump(all_metrics, mf, indent=4)
    print(f"Metrics per recording salvate: {metrics_file}")

    summary = {}
    for method, vals in method_metrics.items():
        if vals:
            arr = np.array(vals)
            summary[method] = {
                'ARI_mean': round(float(np.mean(arr[:,0])), 4),
                'DER_mean': round(float(np.mean(arr[:,1])), 4)
            }
    summary_file = os.path.join(RESULTS_DIR, 'average_metrics_by_method.json')
    with open(summary_file, 'w') as sf:
        json.dump(summary, sf, indent=4)
    print(f"Mediile metricilor salvate: {summary_file}")




