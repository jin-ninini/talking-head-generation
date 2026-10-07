import os
import librosa
import numpy as np
import torch
import torchaudio
from scipy.io import wavfile

# -----------------------------
# STT 모델
# -----------------------------
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# -----------------------------
# MBART 번역 모델
# -----------------------------
from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

# -----------------------------
# FastSpeech2 + VocGAN
# -----------------------------
from . import hparams as hp
from .synthesize import get_FastSpeech2, kor_preprocess, synthesize as fs2_synthesize
from .utils import get_vocgan

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch>=2.6 loads with weights_only=True; the FS2/VocGAN checkpoints also hold numpy scalars
from numpy._core.multiarray import scalar as _np_scalar
torch.serialization.add_safe_globals([(_np_scalar, "numpy.core.multiarray.scalar"), np.dtype,
                                      type(np.dtype(np.float64)), type(np.dtype(np.int64))])

print("🔹 STT / 번역 모델 로드 중...")

# STT
STT_MODEL_NAME = "facebook/wav2vec2-base-960h"
stt_processor = Wav2Vec2Processor.from_pretrained(STT_MODEL_NAME)
stt_model = Wav2Vec2ForCTC.from_pretrained(STT_MODEL_NAME).to(device)

# 번역 (MBART)
MT_MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
mt_tokenizer = MBart50TokenizerFast.from_pretrained(MT_MODEL_NAME)
mt_model = MBartForConditionalGeneration.from_pretrained(MT_MODEL_NAME).to(device)


# ------------------------------------------------
# 1. Speech → English text
# ------------------------------------------------
def load_speech(audio_path: str):
    speech, sr = torchaudio.load(audio_path)

    if speech.size(0) > 1:
        speech = speech.mean(dim=0, keepdim=True)

    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        speech = resampler(speech)
        sr = target_sr
    return speech, sr


def split_on_silence(speech, sr, top_db=30, max_gap=0.3, max_len=10.0):
    # wav2vec2 emits no punctuation, so long input is cut into phrases at pauses
    intervals = librosa.effects.split(speech.squeeze(0).numpy(), top_db=top_db)
    segments = []
    for start, end in intervals:
        if segments and (start - segments[-1][1]) < max_gap * sr and (end - segments[-1][0]) <= max_len * sr:
            segments[-1][1] = end
        else:
            segments.append([start, end])
    return segments


def speech_to_text(speech, sr) -> str:
    print("\n[1] STT 실행")

    inputs = stt_processor(
        speech.squeeze(0),
        sampling_rate=sr,
        return_tensors="pt",
        padding=True
    )

    with torch.no_grad():
        logits = stt_model(inputs.input_values.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    text = stt_processor.batch_decode(predicted_ids)[0]
    text = text.strip()
    print("  ➡ STT 결과(영어):", text)
    return text

# ------------------------------------------------
# 2. English → Korean 
# ------------------------------------------------
def translate_to_korean(english_text):
    print("🌐 [Translate] 영어 → 한국어 번역 중...")

    mt_tokenizer.src_lang = "en_XX"

    encoded = mt_tokenizer(english_text, return_tensors="pt").to(device)  

    generated_tokens = mt_model.generate(
        **encoded,
        forced_bos_token_id=mt_tokenizer.lang_code_to_id["ko_KR"]
    )

    korean_text = mt_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print("➡ 번역 결과:", korean_text)
    return korean_text



# -----------------------------
# 3. Korean text → Speech (FastSpeech2 + VocGAN)
# -----------------------------
def load_tts():
    # FastSpeech2 모델 로드 (사전학습 checkpoint step 사용)
    step = 350000  # checkpoint_350000.pth.tar
    print(f"  ▸ FastSpeech2 checkpoint step = {step}")
    model = get_FastSpeech2(step)
    model = model.to(device)
    model.eval()

    # VocGAN 로드
    vocoder = get_vocgan(hp.vocoder_pretrained_model_path)
    return model, vocoder


def tts_korean(korean_text: str, model, vocoder, prefix: str) -> str:
    print("\n[3] TTS 실행 (한국어 텍스트 → 음성)")

    # 한글 전처리 + 텍스트 텐서 생성
    text_tensor, src_len = kor_preprocess(korean_text)  

    # 기본값 설정 (None 처리 방지)
    dur_pitch_energy_aug = [1.0, 1.0, 1.0]  

    # 음성 합성 (파일은 hp.test_path 아래에 저장됨)
    fs2_synthesize(
        model,
        vocoder,
        text_tensor,
        korean_text,
        prefix=prefix,  
        dur_pitch_energy_aug=dur_pitch_energy_aug,  
        src_len=src_len  
    )
    print(f"  ➡ 합성 완료! {hp.test_path} 폴더에 wav 파일이 생성됩니다.")
    return os.path.join(hp.test_path, '{}_{}_{}.wav'.format(prefix, hp.vocoder, korean_text[:10]))


# ------------------------------------------------
# 4. Full PIPELINE
# ------------------------------------------------
def run_pipeline(input_wav, out_dir, pause=0.3):
    print("\n🚀 전체 파이프라인 시작\n")

    hp.test_path = os.path.join(out_dir, "segments")
    model, vocoder = load_tts()

    speech, sr = load_speech(input_wav)
    segments = split_on_silence(speech, sr)
    silence = np.zeros(int(pause * hp.sampling_rate), dtype=np.int16)
    lines, waves = [], []
    for i, (start, end) in enumerate(segments):
        english = speech_to_text(speech[:, start:end], sr)
        if not english:
            continue
        korean = translate_to_korean(english)
        wav_path = tts_korean(korean, model, vocoder, prefix=f"{i:02d}")
        waves += [wavfile.read(wav_path)[1], silence]
        lines.append(f"{i:02d}\t{start / sr:.2f}-{end / sr:.2f}\t{english}\t{korean}")

    wavfile.write(os.path.join(out_dir, "korean.wav"), hp.sampling_rate, np.concatenate(waves))
    with open(os.path.join(out_dir, "transcript.tsv"), "w", encoding="utf-8") as f:
        f.write("segment\tsource_sec\tenglish\tkorean\n" + "\n".join(lines) + "\n")

    print("\n✅ 파이프라인 완료")

