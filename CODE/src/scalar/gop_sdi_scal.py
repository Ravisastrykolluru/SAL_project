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


def calculate_numerator(probabilities, canonical_transcription):
    num_frames = probabilities.shape[1]
    alphas = torch.full((len(canonical_transcription) + 1, num_frames + 1), -float('inf'), dtype=torch.float64)
    alphas[0, 0] = 0  # Start state

    for t in range(1, num_frames + 1):
        for i, phoneme in enumerate(canonical_transcription):
            phoneme_id = phoneme_to_id.get(phoneme, None)
            if phoneme_id is None or phoneme_id >= probabilities.shape[2]:
                continue  # Skip invalid phoneme IDs

            log_prob = torch.log(torch.clamp(probabilities[0, t - 1, phoneme_id], min=1e-10))
            alphas[i + 1, t] = torch.logsumexp(
                torch.stack([
                    alphas[i, t - 1] + log_prob,  # Transition from the previous state
                    alphas[i + 1, t - 1]  # Staying in the current state
                ]),
                dim=0
            )

    return torch.exp(alphas[len(canonical_transcription), num_frames]).item()

def calculate_denominator(probabilities, canonical_transcription, target_phoneme):
    num_frames = probabilities.shape[1]
    total_probability = 0.0

    remaining_phonemes = [phoneme for phoneme in phoneme_to_id.keys() if phoneme != target_phoneme]
    if not remaining_phonemes:
        return total_probability  

    substitute_phoneme = remaining_phonemes[0] 

    # Substitution
    substituted_transcription = [
        p if p != target_phoneme else substitute_phoneme
        for p in canonical_transcription
    ]
    total_probability += calculate_numerator(probabilities, substituted_transcription)

    # Insertion
    for i in range(len(canonical_transcription) + 1):
        inserted_transcription = (
            canonical_transcription[:i] + [substitute_phoneme] + canonical_transcription[i:]
        )
        total_probability += calculate_numerator(probabilities, inserted_transcription)

    # Deletion
    for i, phoneme in enumerate(canonical_transcription):
        if phoneme == target_phoneme:
            deleted_transcription = canonical_transcription[:i] + canonical_transcription[i + 1:]
            total_probability += calculate_numerator(probabilities, deleted_transcription)

    return total_probability


input_file = "path to files.scp"
output_file = "text file path to save GOP scores"

with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        audio_path, canonical_transcription_str = line.strip().split('\t')
        canonical_transcription = canonical_transcription_str.split()


        filename = os.path.basename(audio_path)

     
        waveform, sample_rate = preprocess_audio(audio_path)

        inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(**inputs).logits

        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        model_vocab_size = probabilities.shape[2]
        phoneme_to_id = {token: idx for idx, token in enumerate(processor.tokenizer.get_vocab()) if idx < model_vocab_size}

        LPR_scores = []
        numerator = calculate_numerator(probabilities, canonical_transcription)

        for phoneme in canonical_transcription:
            denominator = calculate_denominator(probabilities, canonical_transcription, phoneme)
            if denominator > 0:
                LPR = np.log(numerator / denominator)
            else:
                LPR = float('-inf')
            LPR_scores.append((phoneme, LPR))

        outfile.write(f"{filename}\t")
        outfile.write(", ".join([f"{phoneme}:{score:.4f}" for phoneme, score in LPR_scores]))
            # outfile.write(f"{phoneme}: {score:.4f}\t")
        outfile.write("\n")

print(f"GOP values written to {output_file_path}")
