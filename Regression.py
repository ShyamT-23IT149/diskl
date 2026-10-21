import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

df = pd.read_csv("1553768847-housing.csv")

df = pd.get_dummies(df,drop_first=True)
df.fillna(df.mean(numeric_only=True),inplace=True)


X = df.drop("median_house_value",axis=1)
Y = df["median_house_value"]

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

print(mean_squared_error(Y_test,y_pred))
print(r2_score(Y_test,y_pred))