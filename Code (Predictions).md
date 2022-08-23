```
from sklearn.model_selection import train_test_split 
X = df[[]]
y = df[]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```
```
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression(solver='lbfgs')
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
```
```
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))
```
```
from sklearn.metrics import confusion_matrix
# Printing the confusion_matrix
print(confusion_matrix(y_test, predictions))
```
