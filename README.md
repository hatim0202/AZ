1. Linear Regression (House price vs area)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.DataFrame({
    'area':  [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750],
    'price': [25, 36, 49, 60, 74, 85, 99, 110, 124, 135]   # lakhs
})
X = df[['area']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted:", y_pred)
print("Coef:", model.coef_, "Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))
print("Price for 1800 sqft:", model.predict([[1800]]))

2. Multivariate Linear Regression (area + rooms, price)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.DataFrame({
    'area':  [500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750],
    'rooms': [1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
    'price': [26, 38, 48, 63, 72, 83, 101, 112, 128, 137]
})
X = df[['area', 'rooms']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted:", y_pred)
print("Coef:", model.coef_, "Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 :", r2_score(y_test, y_pred))
print("Price (1600 sqft, 3 rooms):", model.predict([[1600, 3]]))

3.Logistic Regression (Hours studied, Pass/Fail)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.DataFrame({
    'hours':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'attend': [50, 55, 60, 58, 65, 80, 85, 90, 88, 95],
    'result': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]   # 1 = pass
})
X = df[['hours', 'attend']]
y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Student (6 hrs, 75%):", model.predict([[6, 75]]))

4.SVM (Age, Salary, Buys product or not)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.DataFrame({
    'age':    [22, 25, 28, 30, 35, 40, 45, 48, 52, 55],
    'salary': [15, 18, 20, 25, 40, 55, 60, 70, 80, 90],   # k per month
    'buy':    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})
X = df[['age', 'salary']]
y = df['buy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Customer (38, 50):", model.predict([[38, 50]]))

5.Bagging (Loan approval)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.DataFrame({
    'income': [20, 25, 30, 35, 40, 50, 60, 70, 80, 90],
    'credit': [450, 500, 520, 550, 600, 650, 700, 720, 750, 800],
    'loan':   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]   # 1 = approved
})
X = df[['income', 'credit']]
y = df['loan']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = BaggingClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Applicant (55, 680):", model.predict([[55, 680]]))

6.Boosting (Spam detection)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.DataFrame({
    'links':  [0, 1, 0, 1, 2, 5, 6, 7, 8, 9],
    'caps':   [2, 3, 1, 4, 5, 20, 25, 30, 28, 35],   # CAPS words count
    'spam':   [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
})
X = df[['links', 'caps']]
y = df['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = AdaBoostClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Predicted:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Mail (4 links, 22 caps):", model.predict([[4, 22]]))
