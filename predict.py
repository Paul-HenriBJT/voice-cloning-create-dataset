#!/usr/bin/env python3

# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path as CogPath
from pathlib import Path
import os
import shutil
import subprocess
import numpy as np
import librosa
import soundfile


# This function is obtained from librosa.
def get_rms(
    y,
    *,
    frame_length=2048,
    hop_length=512,
    pad_mode="constant",
):
    padding = (int(frame_length // 2), int(frame_length // 2))
    y = np.pad(y, padding, mode=pad_mode)

    axis = -1
    out_strides = y.strides + tuple([y.strides[axis]])
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)


class Slicer:
    def __init__(
        self,
        sr: int,
        threshold: float = -40.0,
        min_length: int = 5000,
        min_interval: int = 300,
        hop_size: int = 20,
        max_sil_kept: int = 5000,
    ):
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[
                :, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)
            ]
        else:
            return waveform[
                begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)
            ]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return [waveform]
        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None:
                    silence_start = i
                continue
            if silence_start is None:
                continue
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                i - silence_start >= self.min_interval
                and i - clip_start >= self.min_length
            )
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = (
                    rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                    ].argmin()
                    + silence_start
                )
                pos_r = (
                    rms_list[i - self.max_sil_kept : i + 1].argmin()
                    + i
                    - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        total_frames = rms_list.shape[0]
        if (
            silence_start is not None
            and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
                )
            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    self._apply_slice(waveform, sil_tags[-1][1], total_frames)
                )
            return chunks


#!/usr/bin/env python3

# Previous imports remain the same...
from cog import BasePredictor, Input, Path as CogPath
from pathlib import Path
# ... other imports remain the same ...

class Predictor(BasePredictor):
    def setup(self) -> None:
        pass

    def predict(
        self,
        audio_file: CogPath = Input(
            description="Audio file (MP3 or WAV) to create your RVC v2 dataset from",
        ),
        audio_name: str = Input(
            default="rvc_v2_voices",
            description="Name of the dataset. The output will be a zip file containing a folder named `dataset/<audio_name>/`. This folder will include multiple `.wav` files named as `split_<i>.wav`. Each `split_<i>.wav` file is a short audio clip with isolated vocals.",
        ),
    ) -> CogPath:
        """Run a single prediction on the model"""
        
        AUDIO_NAME = audio_name

        # Empty old folders
        folders = [
            "separated/htdemucs",
            f"dataset/{AUDIO_NAME}",
        ]
        for folder in folders:
            try:
                shutil.rmtree(folder)
            except FileNotFoundError:
                pass

        # Delete old output
        try:
            os.remove(f"dataset_{AUDIO_NAME}.zip")
        except FileNotFoundError:
            pass

        # Create necessary directories
        os.makedirs("separated/htdemucs", exist_ok=True)
        os.makedirs(f"dataset/{AUDIO_NAME}", exist_ok=True)

        # Convert input to WAV if it's MP3
        if str(audio_file).lower().endswith('.mp3'):
            output_wav = "input_audio.wav"
            command = f"ffmpeg -i {audio_file} {output_wav}"
            subprocess.run(command.split(), check=True)
            input_path = output_wav
        else:
            input_path = str(audio_file)

        # Separate Vocal and Instrument/Noise using Demucs
        command = f"demucs --two-stems=vocals {input_path}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, _ = process.communicate()
        print(output.decode())

        # Get the base name of the input file without extension
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Load and process the vocals
        vocals_path = f"separated/htdemucs/{base_name}/vocals.wav"
        audio, sr = librosa.load(vocals_path, sr=None, mono=False)
        
        slicer = Slicer(
            sr=sr,
            threshold=-40,
            min_length=5000,
            min_interval=500,
            hop_size=10,
            max_sil_kept=500,
        )
        chunks = slicer.slice(audio)
        
        # Save the sliced audio files
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T
            soundfile.write(f"dataset/{AUDIO_NAME}/split_{i}.wav", chunk, sr)

        # Create zip file
        output_zip_path = f"dataset_{AUDIO_NAME}.zip"
        audio_folder_path = Path("dataset") / AUDIO_NAME
        
        zip_command = [
            "zip",
            "-r",
            output_zip_path,
            audio_folder_path.as_posix(),
            "-i",
            f"{audio_folder_path.as_posix()}/*",
        ]

        with subprocess.Popen(
            zip_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=Path.cwd(),
        ) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                print(f"Error in zipping: {stderr.decode()}")
            else:
                print(f"Output: {stdout.decode()}")

        # Cleanup
        if str(audio_file).lower().endswith('.mp3'):
            try:
                os.remove(output_wav)
            except FileNotFoundError:
                pass

        return CogPath(output_zip_path)