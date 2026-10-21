import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("1553768847-housing.csv")
#print(df.info())

df = df.select_dtypes(include=["float64","int64"])
#df = pd.get_dummies(df,columns=["ocean_proximity"],drop_first=True)

print("Mean: \n")
print(df.mean())


print("Median: \n")
print(df.median())

print(df.mode())


print(df.max() - df.min())
print(df.var())
print(df.std())




sns.histplot(df["median_income"],kde=True,color="blue")
plt.title("California House Price")
plt.show()
