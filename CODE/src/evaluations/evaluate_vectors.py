
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split


actual_scores_df = pd.read_csv("../../data/metadata.csv")
generated_vectors_df = pd.read_csv("path to saved vectors csv")

actual_scores_df['file_name'] = actual_scores_df['file_name'].str.lower()
generated_vectors_df['Audio File'] = generated_vectors_df['Audio File'].str.lower()

generated_vectors_df["Vector Sequence"] = generated_vectors_df["Vector Sequence"].apply(eval)

actual_scores_df['p_scores'] = actual_scores_df['p_scores'].apply(lambda x: list(map(float, x.split())))

merged_df = pd.merge(
    actual_scores_df,
    generated_vectors_df,
    left_on='file_name',
    right_on='Audio File',
    how='inner'
)

print(len(merged_df['p_scores'][0]))
print(merged_df['p_scores'][0])


max_length = 48  

def pad_sub_arrays(seq, length):
    return [sub_array + [0] * (length - len(sub_array)) if len(sub_array) < length else sub_array[:length] for sub_array in seq]

merged_df['Vector Sequence'] = merged_df['Vector Sequence'].apply(
    lambda seq: pad_sub_arrays(seq, max_length)
)


X = np.vstack(merged_df['Vector Sequence'].values)
y = np.hstack(merged_df['p_scores'].values)
print(f"X shape: {X.shape}, y shape: {y.shape}")


X = []
y = []

for _, row in merged_df.iterrows():
    vectors = row["Vector Sequence"]
    scores = row["p_scores"]  
    
    if len(vectors) == len(scores):
        X.extend(vectors)
        y.extend(scores)

X = np.array(X)
y = np.array(y)

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")



# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svr_model = SVR(kernel="rbf", C=1.0, epsilon=0.1)
svr_model.fit(X, y)

y_train_pred = svr_model.predict(X[:30])

pcc, _ = pearsonr(y, y_train_pred)

print("Pearson Correlation Coefficient (PCC):", np.abs(pcc))

# results_df = pd.DataFrame({"True Scores": y_test, "Predicted Scores": y_pred})
# results_df.to_csv("svr_results.csv", index=False)
