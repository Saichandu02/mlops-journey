from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

data=load_iris()

X_train,X_test,y_train,y_test = train_test_split(data.data, data.target)
model=RandomForestClassifier()
model.fit(X_train,y_train)
print(f"Accuracy: {model.score(X_test, y_test):.2f}")
with open("model.pkl","wb") as f:
    pickle.dump(model, f)
