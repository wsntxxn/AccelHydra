import hashlib
import tempfile

from pydub import AudioSegment, silence
from librosa.filters import mel as librosa_mel_fn
import torch
import torch.nn as nn
import torchaudio
from transformers import pipeline

# raw wav to mel spec

mel_basis_cache = {}
hann_window_cache = {}

# load asr pipeline

asr_pipe = None


def get_bigvgan_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
    fmin=0,
    fmax=None,
    center=False,
):  # Copy from https://github.com/NVIDIA/BigVGAN/tree/main
    device = waveform.device
    key = f"{n_fft}_{n_mel_channels}_{target_sample_rate}_{hop_length}_{win_length}_{fmin}_{fmax}_{device}"

    if key not in mel_basis_cache:
        mel = librosa_mel_fn(
            sr=target_sample_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=fmin,
            fmax=fmax
        )
        mel_basis_cache[key] = torch.from_numpy(mel).float().to(
            device
        )  # TODO: why they need .float()?
        hann_window_cache[key] = torch.hann_window(win_length).to(device)

    mel_basis = mel_basis_cache[key]
    hann_window = hann_window_cache[key]

    padding = (n_fft - hop_length) // 2
    waveform = torch.nn.functional.pad(
        waveform.unsqueeze(1), (padding, padding), mode="reflect"
    ).squeeze(1)

    spec = torch.stft(
        waveform,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec


def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


class MelSpec(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    ):
        super().__init__()
        assert mel_spec_type in [
            "vocos", "bigvgan"
        ], print("We only support two extract mel backend: vocos or bigvgan")

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        if mel_spec_type == "vocos":
            self.extractor = get_vocos_mel_spectrogram
        elif mel_spec_type == "bigvgan":
            self.extractor = get_bigvgan_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel


def remove_silence_edges(audio, silence_threshold=-42):
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(
        audio, silence_threshold=silence_threshold
    )
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[:int(non_silent_end_duration * 1000)]

    return trimmed_audio


def initialize_asr_pipeline(dtype=None):
    if dtype is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = (
            torch.float16 if "cuda" in device and
            torch.cuda.get_device_properties(device).major >= 7 and
            not torch.cuda.get_device_name().endswith("[ZLUDA]") else
            torch.float32
        )
    global asr_pipe
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        torch_dtype=dtype,
        device=device,
    )


def whisper_transcribe(ref_audio, language=None):
    global asr_pipe
    if asr_pipe is None:
        initialize_asr_pipeline()
    return asr_pipe(
        ref_audio,
        chunk_length_s=30,
        batch_size=128,
        generate_kwargs={
            "task": "transcribe",
            "language": language
        } if language else {"task": "transcribe"},
        return_timestamps=False,
    )["text"].strip()


def preprocess_ref_audio_text(ref_audio_orig, ref_text, show_info=print):
    show_info("Converting audio...")

    # Compute a hash of the reference audio file
    with open(ref_audio_orig, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_hash = hashlib.md5(audio_data).hexdigest()

    # if audio_hash in _ref_audio_cache:
    # show_info("Using cached preprocessed reference audio...")
    # ref_audio = _ref_audio_cache[audio_hash]

    # else:  # first pass, do preprocess

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    aseg = AudioSegment.from_file(ref_audio_orig)

    # 1. try to find long silence for clipping
    non_silent_segs = silence.split_on_silence(
        aseg,
        min_silence_len=1000,
        silence_thresh=-50,
        keep_silence=1000,
        seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        if len(non_silent_wave
              ) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
            show_info("Audio is over 12s, clipping short. (1)")
            break
        non_silent_wave += non_silent_seg

    # 2. try to find short silence for clipping if 1. failed
    if len(non_silent_wave) > 12000:
        non_silent_segs = silence.split_on_silence(
            aseg,
            min_silence_len=100,
            silence_thresh=-40,
            keep_silence=1000,
            seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave
                  ) > 6000 and len(non_silent_wave + non_silent_seg) > 12000:
                show_info("Audio is over 12s, clipping short. (2)")
                break
            non_silent_wave += non_silent_seg

    aseg = non_silent_wave

    # 3. if no proper silence found for clipping
    if len(aseg) > 12000:
        aseg = aseg[:12000]
        show_info("Audio is over 12s, clipping short. (3)")

    aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
    aseg.export(temp_path, format="wav")
    ref_audio = temp_path

    if not ref_text.strip():
        show_info(
            "No reference text provided, transcribing reference audio..."
        )
        ref_text = whisper_transcribe(ref_audio)
        # Cache the transcribed text (not caching custom ref_text, enabling users to do manual tweak)
    else:
        show_info("Using custom reference text...")

    # Ensure ref_text ends with a proper sentence-ending punctuation
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "

    print("\nref_text  ", ref_text)

    return ref_audio, ref_text
