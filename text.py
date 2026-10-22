import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Load text file
df = pd.read_csv("dataset.txt", encoding="utf-8-sig")

# Standardize column headers
df.columns = df.columns.astype(str).str.strip().str.lower()

text_col = "sanitized_property_summary"
target_col = "type"

df = df.dropna(subset=[text_col, target_col])

X = df[text_col]
y = df[target_col]

# Use stratify=y to ensure all classes are present in train and test splits
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Vectorization
tfidf = TfidfVectorizer()
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",classification_report(y_test, y_pred, zero_division=0),)
