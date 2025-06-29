import librosa
import numpy as np

def process_audio_to_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    spec = librosa.feature.melspectrogram(y=y, sr=sr)
    spec_db = librosa.power_to_db(spec, ref=np.max)
    spec_db = np.resize(spec_db, (128, 128, 1))
    return np.expand_dims(spec_db, axis=0)
