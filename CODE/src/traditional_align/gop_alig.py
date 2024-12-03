import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import numpy as np
import os


checkpoint_dir = "path to load model "
processor = Wav2Vec2Processor.from_pretrained(checkpoint_dir)
model = Wav2Vec2ForCTC.from_pretrained(checkpoint_dir)
model.eval()

def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform.squeeze(), 16000

def calculate_gop(audio_path, canonical_sequence):
    waveform, sample_rate = preprocess_audio(audio_path)
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    vocab = processor.tokenizer.get_vocab()
    phoneme_to_id = {phoneme: idx for phoneme, idx in vocab.items()}

    gop_scores = []
    for frame_idx, frame_probs in enumerate(probabilities[0]):
        if frame_idx >= len(canonical_sequence): 
            break

        target_phoneme = canonical_sequence[frame_idx]
        target_phoneme_id = phoneme_to_id.get(target_phoneme, None)

        if target_phoneme_id is None:
            print(f"Phoneme {target_phoneme} not found in the vocabulary.")
            continue

        p_target = frame_probs[target_phoneme_id].item()

        p_non_target = frame_probs.sum().item() - p_target

        if p_non_target > 0:
            gop = np.log(p_target / p_non_target)
        else:
            gop = float("-inf")  
        gop_scores.append((target_phoneme, gop))

    return gop_scores

def process_files(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue

            audio_path, phoneme_sequence = parts
            canonical_sequence = phoneme_sequence.split()

            gop_scores = calculate_gop(audio_path, canonical_sequence)
            # print(gop_scores)
           
            filename = os.path.basename(audio_path)

            outfile.write(f"{filename}\t")
            outfile.write(", ".join([f"{phoneme}:{score:.4f}" for phoneme, score in gop_scores]))
            outfile.write("\n")

input_file = "path to files.scp"
output_file = "text file path to save GOP scores"

process_files(input_file, output_file)
