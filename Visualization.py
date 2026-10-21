import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("1553768847-housing.csv")

#df["median_house_value"].hist()
sns.histplot(df["median_house_value"],kde=True)
plt.title("Median House Value")
plt.show()

sns.boxplot(x=df["median_house_value"])
plt.title("A")
plt.show()

sns.countplot(x="ocean_proximity",data=df)
plt.title("Count Plot")
plt.show()


sns.barplot(x="ocean_proximity",y="median_house_value",data=df)
plt.title("Bar Plot")
plt.show()


sns.scatterplot(x="median_income",y="median_house_value",data=df)
plt.title("Scatter Plot")
plt.show()

sns.pairplot(df) 
plt.show()

sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.title("Heatmap")
plt.show()