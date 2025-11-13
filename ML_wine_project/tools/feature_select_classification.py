# feature_select_classification.py
# Usage (change paths/columns below and just click Run in PyCharm)

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# --- EDIT THESE ---
CSV_PATH = "data/raw_data/winequality-white.csv"
LABEL = "quality"              # your categorical label
DROP_ALWAYS = ["", LABEL]      # columns to drop from features
K = 20                          # how many top features to show
# -------------------

# Load
df = pd.read_csv(CSV_PATH)

# Basic cleaning: keep only numeric features for f_classif (or one-hot encode before)
X = df.drop(columns=DROP_ALWAYS, errors="ignore").select_dtypes(include="number")
y = df[LABEL]

# Select top-K
selector = SelectKBest(score_func=f_classif, k=min(K, X.shape[1]))
X_new = selector.fit_transform(X, y)
scores = pd.DataFrame({"Feature": X.columns, "Score": selector.scores_})

print("\nTop features (classification, ANOVA F):")
print(scores.sort_values("Score", ascending=False).head(K).to_string(index=False))

