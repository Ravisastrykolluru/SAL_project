import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
import numpy as np

checkpoint_dir = "path to load the model"

processor = Wav2Vec2Processor.from_pretrained(checkpoint_dir)
model = Wav2Vec2ForCTC.from_pretrained(checkpoint_dir)

model.eval()

def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform.squeeze(), 16000

def calculate_numerator(probabilities, canonical_transcription):
    num_frames = probabilities.shape[1]
    alphas = torch.full((len(canonical_transcription) + 1, num_frames + 1), -float('inf'), dtype=torch.float64)
    alphas[0, 0] = 0  

    for t in range(1, num_frames + 1):
        for i, phoneme in enumerate(canonical_transcription):
            phoneme_id = phoneme_to_id.get(phoneme, None)
            if phoneme_id is None or phoneme_id >= probabilities.shape[2]:
                continue 

            log_prob = torch.log(torch.clamp(probabilities[0, t - 1, phoneme_id], min=1e-10))
            alphas[i + 1, t] = torch.logsumexp(
                torch.stack([
                    alphas[i, t - 1] + log_prob,  # Transition from the previous state
                    alphas[i + 1, t - 1]  # Staying in the current state
                ]),/home/lenovo/Music/Project_temp/vectors_evlauation/generate.py
                dim=0
            )

    return torch.exp(alphas[len(canonical_transcription), num_frames]).item()

def calculate_modifications(probabilities, canonical_transcription, target_phoneme):
    num_frames = probabilities.shape[1]
    total_probability = []

    phonemes = list(phoneme_to_id.keys())
    remaining_phonemes = [phonemes[i] for i in range(0, 39)]

    for substitute_phoneme in remaining_phonemes:
        substituted_transcription = [
            p if p != target_phoneme else substitute_phoneme
            for p in canonical_transcription
        ]
        total_probability.append(calculate_numerator(probabilities, substituted_transcription))

    for i, phoneme in enumerate(canonical_transcription):
        if phoneme == target_phoneme:
            deleted_transcription = canonical_transcription[:i] + canonical_transcription[i + 1:]
            total_probability.append(calculate_numerator(probabilities, deleted_transcription))

    return total_probability

input_file_path = "files.scp"
output_dir = "path to save vectors text files"

with open(input_file_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    audio_path, canonical_transcription = line.strip().split('\t')
    canonical_transcription = canonical_transcription.split()

    waveform, sample_rate = preprocess_audio(audio_path)

    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    model_vocab_size = probabilities.shape[2]
    phoneme_to_id = {token: idx for idx, token in enumerate(processor.tokenizer.get_vocab()) if idx < model_vocab_size}

    numerator = calculate_numerator(probabilities, canonical_transcription)
    print(numerator)

    
    feature_vectors = []
    for phoneme in canonical_transcription:
        modifications_probabilities = calculate_modifications(probabilities, canonical_transcription, phoneme)
        feature_vector = [numerator] + modifications_probabilities
        feature_vectors.append((phoneme, feature_vector))


    output_file_path = f"{output_dir}{audio_path.split('/')[-1].split('.')[0]}_feature_vectors.txt"
    with open(output_file_path, 'w') as output_file:
        for phoneme, vector in feature_vectors:
            output_file.write(f"{phoneme}: {vector}\n")

    print(f"Feature vectors for {audio_path} written to {output_file_path}")
