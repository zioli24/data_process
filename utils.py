from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import torchaudio
import torchaudio.transforms as T


def resample(file, target_sr):
    wav, sr = torchaudio.load(file)
    if target_sr:
        resampler = T.Resample(sr, target_sr, dtype=wav.dtype)
        resampled_waveform = resampler(wav)
        torchaudio.save(file, resampled_waveform, target_sr)


def resample_batch(in_dir, target_sr):
    futures = []
    with ProcessPoolExecutor(max_workers=16) as executor:
        for wavfile in tqdm(list(Path(in_dir).rglob("*.wav")), desc="add to pool"):
            futures.append(executor.submit(resample, wavfile, target_sr))
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="norm"):
            future.result()


    
