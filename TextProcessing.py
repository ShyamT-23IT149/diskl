from sklearn.feature_extraction.text import CountVectorizer
texts = [
    "I love data science",
    "Machine learning is powerful",
    "I love machine learning",
    "Data science is interesting"
]
vectorizer = CountVectorizer()
transformed_text = vectorizer.fit_transform(texts)
print(vectorizer.get_feature_names_out())
print(transformed_text.toarray())