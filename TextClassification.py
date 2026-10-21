from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

texts = [
    "I love this product",
    "This is amazing",
    "I hate this",
    "This is bad",
    "Very good experience",
    "Worst service ever"
]

labels = [1, 1, 0, 0, 1, 0] 


vect = CountVectorizer()
X = vect.fit_transform(texts)


X_train , X_test , y_train , y_test = train_test_split(X,labels,test_size=0.2,random_state=42)
model = MultinomialNB()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)


print(accuracy_score(y_test,y_pred))
