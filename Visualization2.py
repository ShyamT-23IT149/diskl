import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file = "sample_data.xlsx"

df = pd.read_csv(file) if file.endswith(".csv") else pd.read_excel(file)

num = df.select_dtypes(include="number")
cat = df.select_dtypes(include=["object", "category"])

print(df.head())
print("Numeric columns:", num.columns.tolist())
print("Categorical columns:", cat.columns.tolist())

# 1. Histogram
if len(num.columns) > 0:
    sns.histplot(data=num, x=num.columns[0], kde=True)
    plt.title("Histogram")
    plt.show()

# 2. Box Plot
if len(num.columns) > 0:
    sns.boxplot(x=num[num.columns[0]])
    plt.title("Box Plot")
    plt.show()

# 3. Count Plot
if len(cat.columns) > 0:
    sns.countplot(data=df, x=cat.columns[0])
    plt.xticks(rotation=45)
    plt.title("Count Plot")
    plt.tight_layout()
    plt.show()

# 4. Bar Plot
if len(cat.columns) > 0 and len(num.columns) > 0:
    sns.barplot(data=df, x=cat.columns[0], y=num.columns[0])
    plt.xticks(rotation=45)
    plt.title("Bar Plot")
    plt.tight_layout()
    plt.show()

# 5. Scatter Plot
if len(num.columns) >= 2:
    sns.scatterplot(data=df, x=num.columns[0], y=num.columns[1])
    plt.title("Scatter Plot")
    plt.show()

# 6. Pair Plot
if len(num.columns) >= 2:
    sns.pairplot(num.iloc[:, :5])
    plt.show()

# 7. Correlation Heatmap
if len(num.columns) >= 2:
    sns.heatmap(num.corr(), annot=True, cmap="coolwarm")
    plt.title("Heatmap")
    plt.show()
