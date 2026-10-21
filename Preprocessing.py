import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("1553768847-housing.csv")


print(df.isnull().sum())


df.fillna(df.mean(numeric_only=True),inplace=True)

print(df.isnull().sum())

df = pd.get_dummies(df,drop_first=True)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

print(scaled_data[:5])