import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor



df = pd.read_csv("school_progress.csv")
#1
y = df["status"]
f = ["grade","attempts","hours_spent"]
x = df[f]
print(x.shape)
print(y.shape)
# sns.countplot(data=df, x="status")
#2
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(x,y,groups=df["student_id"]))
print(len(set(df["student_id"].iloc[train_idx]) & set(df["student_id"].iloc[test_idx])) > 0)
x_train,x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(pd.DataFrame({"actual":y_test, "predict":predictions}).head(10))
#3
train_predictions = model.predict(x_train)
accuracy3 = [accuracy_score(y_test, predictions), accuracy_score(y_train, train_predictions)]
print(accuracy3[0])
print(sum(y_test==predictions), sum(y_test!=predictions))
ok = y_train.value_counts().idxmax()
print(len(y_train[y_train==ok])/len(y_train))
accuracy_tmp = accuracy_score(y_test, [ok]*len(y_test))
print(accuracy_tmp)
print(accuracy3[0] - accuracy_tmp)
#5
model = DecisionTreeClassifier(max_depth=1, random_state=42)
model.fit(x_train, y_train)
train_predictions = model.predict(x_train)
predictions = model.predict(x_test)
accuracy = [accuracy_score(y_test, predictions), accuracy_score(y_train, train_predictions)]
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(x_train, y_train)
train_predictions = model.predict(x_train)
predictions = model.predict(x_test)
accuracy5 = [accuracy_score(y_test, predictions), accuracy_score(y_train, train_predictions)]
print(pd.DataFrame({"depth":[1, 3, 5],"train":[accuracy[1], accuracy3[1], accuracy5[1]],"test":[accuracy[0], accuracy3[0], accuracy5[0]]}))
sns.barplot(data=pd.DataFrame({"depth":[1,1,3,3,5,5], "accuracy":[accuracy[0], accuracy[1], accuracy3[0], accuracy3[1], accuracy5[0], accuracy5[1]], "group":["test","train"]*3}), x="depth", y="accuracy", hue="group")
#4
data = df[["grade", "attempts", "hours_spent", "lessons_completed", "average_score", "student_id"]].dropna(subset=["average_score"])
f = ["grade", "attempts", "hours_spent", "lessons_completed"]
x = data[f]
y = data["average_score"]
train_idx, test_idx = next(splitter.split(x,y,groups=data["student_id"]))
x_train,x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(pd.DataFrame({"actual":y_test.values, "predict":predictions}).head(10))
mae = mean_absolute_error(y_test, predictions)
print(mae)

plt.show()
