import os
import csv
import ast


text_folder = 'path to vectors text folder'
output_csv = 'path to save vectors csv '


with open(output_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    csvwriter.writerow(['Audio File', 'Phoneme Sequence', 'Vector Sequence'])

  
    for text_file in os.listdir(text_folder):
        if text_file.endswith('_feature_vectors.txt'):
            
            audio_file = text_file.replace('_feature_vectors.txt', '.wav')

            text_file_path = os.path.join(text_folder, text_file)
            phoneme_sequence = []
            vector_sequence = []
            with open(text_file_path, 'r') as file:
                for line in file:
                    if ':' in line:
                        phoneme, vector = line.split(':', 1)
                        phoneme = phoneme.strip()
                        vector = vector.strip()

                        try:
                            vector = ast.literal_eval(vector)
                        except (ValueError, SyntaxError):
                            print(f"Error parsing vector for phoneme {phoneme} in file {text_file}")
                            continue

                        phoneme_sequence.append(phoneme)
                        vector_sequence.append(vector)

            csvwriter.writerow([audio_file, ' '.join(phoneme_sequence), vector_sequence])

print(f"CSV file '{output_csv}' created successfully!")
