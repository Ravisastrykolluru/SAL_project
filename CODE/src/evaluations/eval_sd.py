import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

actuals = pd.read_csv("../../data/metadata.csv", header=None)
actuals.columns = ["file_name", "transcription", "p_scores"]

def parse_scores(x):
    try:
        return list(map(float, x.split()))
    except ValueError:
        return None

actuals["p_scores"] = actuals["p_scores"].apply(parse_scores)
actuals = actuals.dropna(subset=["p_scores"])  

predictions = {}
with open("path to saved GOP scores text file - sub,del ", "r") as f:
    for i, line in enumerate(f):
        parts = line.strip().split("\t")
        file_name = parts[0]
        phoneme_scores = parts[1].split(", ")
        phoneme_dict = {p.split(":")[0]: float(p.split(":")[1]) for p in phoneme_scores}
        predictions[file_name] = phoneme_dict

actual_scores = []
predicted_scores = []

for index, row in actuals.iterrows():
    file_name = row["file_name"]
    if file_name in predictions:
        actual_phonemes = row["transcription"].split()
        actual_values = row["p_scores"]

        for phoneme, actual_value in zip(actual_phonemes, actual_values):
            if phoneme in predictions[file_name]:
                actual_scores.append(actual_value)
                predicted_scores.append(predictions[file_name][phoneme])

actual_scores = np.array(actual_scores)
predicted_scores = np.array(predicted_scores)

poly = PolynomialFeatures(degree=2)
predicted_scores_poly = poly.fit_transform(predicted_scores.reshape(-1, 1))

model = LinearRegression()
model.fit(predicted_scores_poly, actual_scores)

predicted_actual_scores = model.predict(predicted_scores_poly)
pcc, _ = pearsonr(actual_scores, predicted_actual_scores)

print(f"Final PCC value: {pcc:.4f}")
