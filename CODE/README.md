# Packages and Datasets
- Download all the packages from given requirements.txt 
- Download speechocean data from "https://www.openslr.org/resources/101/speechocean762.tar.gz" 

# FineTuning

# Data Preperation For FineTuning 
- Create a json file having utterance_id,audio_file_path,phoneme_sequence as attributes for each object (for all the audio files)

- Fine Tune a wav2vec model using librispeech (train.100.clean) with ctc loss such that the last layer should return probabilities for 39 phonemes(standard phoneme set from librispeech kaldi) including a blank token.
 
# Data Preperation For calulation GOP Scores
- Create a files.scp file such that eah line should have audio file path and corresponding phoneme sequence.
- <audio_file_path><tab><phoneme_sequence>

# Generate GOP Scores
- Run all the below codes seperatly using the above obtained files.scp.

```
├── src/
│   ├── tradition_align
│      ├──gop_align.py
│   ├── scalar
│      ├──gop_scalar_sub.py
│      ├──gop_sdi_scal.py
│      ├──gop_sub_del_scalar.py
│   ├── vector_features
│      ├──generate_GOP_vector.py
│      ├──generate_vectors_csv.py      
└── README.md
```

# Evaluation

- Run all the below codes seperatly using the above obtained GOP text files and CSV file.

```
├── src/
│   ├── evaluations
│      ├──eval_align.py
│      ├──eval_sd.py
│      ├──eval_sdi.py 
│      ├──evaluate_vectors.py    
└── README.md
```
