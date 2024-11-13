import os
import argparse
from pathlib import Path
import torch
import soundfile as sf
import numpy as np
import pyloudnorm as pyln
import librosa 
from glob import glob
from tqdm import tqdm
from loguru import logger
from nemo_text_processing.text_normalization.normalize import Normalizer
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def trim_silence(wav):
  """Trim leading and trailing silence

  Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
  """
  # Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. These params are separate and tunable per dataset.
  return librosa.effects.trim(wav, top_db=35,
                              frame_length=512,
                              hop_length=128)[0]
class Slicer:
    def __init__(self,
                 sr: int,
                 threshold: float = -40.,
                 min_length: int = 5000,
                 min_interval: int = 300,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    # This function is obtained from librosa.
    def get_rms(
        self,
        y,
        *,
        frame_length=2048,
        hop_length=512,
        pad_mode="constant",
    ):
        padding = (int(frame_length // 2), int(frame_length // 2))
        y = np.pad(y, padding, mode=pad_mode)

        axis = -1
        # put our new within-frame axis at the end for now
        out_strides = y.strides + tuple([y.strides[axis]])
        # Reduce the shape on the framing axis
        x_shape_trimmed = list(y.shape)
        x_shape_trimmed[axis] -= frame_length - 1
        out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
        xw = np.lib.stride_tricks.as_strided(
            y, shape=out_shape, strides=out_strides
        )
        if axis < 0:
            target_axis = axis - 1
        else:
            target_axis = axis + 1
        xw = np.moveaxis(xw, -1, target_axis)
        # Downsample along the target axis
        slices = [slice(None)] * xw.ndim
        slices[axis] = slice(0, None, hop_length)
        x = xw[tuple(slices)]

        # Calculate power
        power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

        return np.sqrt(power)

    # @timeit
    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if (samples.shape[0] + self.hop_size - 1) // self.hop_size <= self.min_length:
            return [waveform]
        rms_list = self.get_rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))
            if sil_tags[-1][1] < total_frames:
                chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            return chunks
        

class Audio_Process():
    def __init__(self, wav_path=None, gpu='0', sample_rate=24000, min=1.0, max=10.0, normalize=False):
        self.wav_path = wav_path
        self.sample_rate = sample_rate
        self.min = min
        self.max = max
        self.gpu = gpu
        self.normalizer = Normalizer(input_case='cased', lang='en')
        
    def _pyloudnorm(self, audio, sample_rate):
        """
        https://github.com/csteinmetz1/pyloudnorm
        标准: EBU R128 Loudness Normalization and Permitted Maximum Level of Audio Signals
            The Programme Loudness Level shall be normalised to a Target Level of -23.0 LUFS
        """
        # measure the loudness first
        meter = pyln.Meter(sample_rate)  # create BS.1770 meter
        loudness = meter.integrated_loudness(audio)

        # peak normalize audio to -1 dB
        # peak_normalized_audio = pyln.normalize.peak(data, -1.0)  # 貌似是峰值归一化

        # loudness normalize audio to -12 dB LUFS
        loudness_normalized_audio = pyln.normalize.loudness(audio, loudness, -23.0)
        return loudness_normalized_audio

    def trim_add_sil(self, wav_fn):
      sample_rate = librosa.get_samplerate(wav_fn)
      wav,_ = librosa.load(wav, sample_rate)
      wav = trim_silence(wav)
      silen = np.zeros([int(sample_rate * 0.2)])
      wav = np.concatenate([silen, wav, silen], axis=0)
      sf.write(wav_fn, wav, sample_rate)

    def single_audio_process(self, audio_file, save_path, target_sr):
        ori_sr = librosa.get_samplerate(audio_file)
   
        slicer = Slicer(
            sr=ori_sr,
            threshold=-50,
            min_length=3000,
            min_interval=256,
            hop_size=256,
            max_sil_kept=500
        )

        fn = audio_file.split('/')[-1].split('.')[0]
        save_path = os.path.join(save_path, fn)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        audio, _ = librosa.load(audio_file, sr=ori_sr, mono=True)
        audio = self._pyloudnorm(audio, target_sr)  #放后面会有bug
        if ori_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=ori_sr, target_sr=target_sr)

        _chunks = slicer.slice(audio)
        
        #load audios that <1s into former one 
        chunks = []
        for i, seg in enumerate(_chunks):
            if i !=0 and len(seg)/target_sr < 2:
                    seg = np.concatenate([_chunks[i-1], seg])
                    chunks.pop()
                    chunks.append(seg)
            else:
                chunks.append(seg)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T  # Swap axes if the audio is stereo.
            sf.write('{}/{}_{}.wav'.format(save_path, fn, i), chunk, target_sr)  # Save sliced audio files with soundfile.
        logger.info("spliting {} into {} segments".format(audio_file, len(chunks)))    

    def multi_audio_process(self, audio_path, save_path, target_sr, max_workers=96, format='m4a'):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        futures = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for audio_file in tqdm(glob("{}/*.{}".format(audio_path, format)), desc="add to pool"):
                futures.append(
                executor.submit(self.single_audio_process, audio_file, save_path, target_sr)
                )
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()
        return None

    def asr_batch(self, wav_paths, model_id="openai/whisper-large-v3", gpu_id=0):
        device = "cuda:{}".format(gpu_id)
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)
        processor = AutoProcessor.from_pretrained(model_id) 
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=15,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        wav_paths_list = [str(p) for p in Path(wav_paths).iterdir()]
        for wav_path in wav_paths_list:
            wav_dataset = glob(wav_path + '/*.wav')
            logger.info('load asr model successfully, transcripting {} files now..'.format(len(wav_dataset)))
            result = pipe(wav_dataset, generate_kwargs={"language":"english"})
            text = [item["text"] for item in result]
            assert(len(wav_dataset) == len(text))
            with open('{}/transcript.txt'.format(wav_path), 'a+') as f:
                for fn, text in zip(wav_dataset, text):
                    text = self.normalizer.normalize(text, punct_post_process=False)
                    fn = fn.split('/')[-1]
                    line  = "{}|{}".format(fn, text)
                    f.write(line + '\n')
            with open('{}/transcript.txt'.format(wav_path)) as ori:          
                texts = ori.readlines()
                texts.sort(key=lambda x: int(x.split('.')[0].split("_")[-1]))
                with open('{}/transcript.txt'.format(wav_path), 'w') as st:
                    for l in texts:
                        st.write(l)

            logger.info('finish..')
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', type=str)
    parser.add_argument('--savepath', type=str)
    parser.add_argument("--audio_format", type=str, default='m4a')
    parser.add_argument("--target_sr", type=int)
    args = parser.parse_args()
    
    AP = Audio_Process()
    #AP.multi_audio_process(args.audio_path, args.savepath, args.target_sr)
    AP.asr_batch(args.savepath)
    


