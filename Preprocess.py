import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_excel("sample_data.xlsx")
print(df.head())

# Handle missing values
df = df.fillna(df.mean(numeric_only=True))

# Remove duplicates
df = df.drop_duplicates()

# Convert categorical data to numeric
df = pd.get_dummies(df, drop_first=True)

# Save processed data
df.to_csv("processed_dataset2.csv", index=False)

print("Preprocessing completed")

df = pd.read_csv("processed_dataset2.csv")

# 1. Line Plot
plt.plot(df["Employee ID"])
plt.title("Line Plot")
plt.show()

# 2. Bar Plot
df["Salary ($)"].value_counts().plot(kind="bar")
plt.title("Bar Plot")
plt.show()

# 3. Histogram
df["Salary ($)"].plot(kind="hist")
plt.title("Histogram")
plt.show()

# 4. Scatter Plot
plt.scatter(df["Employee ID"], df["Salary ($)"])
plt.title("Scatter Plot")
plt.show()

# 5. Heatmap
corr = df.select_dtypes(include="number").corr()
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.show()
