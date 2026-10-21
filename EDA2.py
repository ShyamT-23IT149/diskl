import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file = "sample_data.xlsx"

if file.endswith(".csv"):
    df = pd.read_csv(file)
else:
    df = pd.read_excel(file)

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())
print("Mean:\n", df.mean(numeric_only=True))
print("Median:\n", df.median(numeric_only=True))
print("Mode:\n", df.mode().iloc[0])
print("Range:\n", df.max(numeric_only=True) - df.min(numeric_only=True))
print("Variance:\n", df.var(numeric_only=True))
print("Standard Deviation:\n", df.std(numeric_only=True))

df.select_dtypes(include="number").hist(figsize=(10, 6))
plt.tight_layout()
plt.show()

sns.heatmap(df.select_dtypes(include="number").corr(), annot=True, cmap="coolwarm")
plt.show()
