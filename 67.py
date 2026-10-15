import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor


df = pd.read_csv("school_progress.csv", index_col="enrollment_id")
#1
y = df["status"]
f = ["grade","attempts","hours_spent"]
x = df[f]
print(x.shape)
print(y.value_counts())
sns.countplot(data=df, x="status")
#2
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(x,y,groups=df["student_id"]))
print(len(set(train_idx) & set(test_idx)) > 0)
x_train,x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
print(pd.DataFrame({"actual":y_test, "predict":predictions}).head(10))
#3
accuracy = accuracy_score(y_test, predictions)
print(accuracy)



# fg,axis= plt.subplots(2,2, figsize=(12,12))
# sns.barplot(data = df, x="grade", y="average_score", hue="level", ax=axis[0,0])
# axis[0,0].set_title("Средний бал по классам и уровням")
# sns.barplot(data = df, x="level", y="average_score", ax=axis[0,1])
# axis[0,1].set_title("Средний бал по уровням")
# sns.barplot(data = df, x="status", y="average_score", hue="level", ax=axis[1,0])
# axis[1,0].set_title("Средний бал по статусу и уровням")
# sns.barplot(data = df, x="grade", y="average_score", ax=axis[1,1])
# axis[1,1].set_title("Средний бал по классам")

plt.show()






