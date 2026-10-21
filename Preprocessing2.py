import pandas as pd
from sklearn.preprocessing import StandardScaler

file = "gdp current.csv"
df = pd.read_csv(file) if file.endswith(".csv") else pd.read_excel(file)
print("Missing Values:\n", df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)
df.fillna(df.mode().iloc[0], inplace=True)

df = pd.get_dummies(df, drop_first=True, dtype=int)
scaled_data = StandardScaler().fit_transform(df)

print("Preprocessed Data:\n", df.head())
print("Scaled Data:\n", scaled_data[:5])
