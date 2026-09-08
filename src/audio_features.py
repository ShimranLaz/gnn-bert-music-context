import librosa
import numpy as np

def extract_log_mel(audio_path, sample_rate=22050, duration=29.0, n_mels=128):
    y, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
    target_len = int(sample_rate * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel_norm = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    return y, sr, log_mel_norm

def extract_chroma_segments(y, sr, segment_number=7):
    seg_len = len(y) // segment_number
    node_features = []
    for i in range(segment_number):
        seg = y[i * seg_len : (i + 1) * seg_len]
        chroma = librosa.feature.chroma_stft(y=seg, sr=sr, n_chroma=12)
        node_features.append(chroma.mean(axis=1))
    return np.array(node_features)
