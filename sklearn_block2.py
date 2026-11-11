import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("school_progress.csv")
x = df[["grade", "attempts", "average_score", "course", "city", "clubs"]]
y=df["status"]
print(x.dtypes)
print(y.dtypes)
print(x.isna().sum())
print(y.isna().sum())
print(x[["course", "city", "clubs"]].dropna().nunique())
#"course", "city", "clubs" в кодировании категорий. clubs avg_score в dropna. все остальые крутые числовые
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(x, y, groups=df["student_id"]))
print(len(set(df["student_id"].iloc[train_idx]) & set(df["student_id"].iloc[test_idx])) > 0)
x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
imputer = SimpleImputer(strategy="median")
x_train["average_score"] = imputer.fit_transform(x_train[["average_score"]])
x_test["average_score"] = imputer.transform(x_test[["average_score"]])
imputer = SimpleImputer(strategy="constant", fill_value="не указано")
x_train[["clubs"]] = imputer.fit_transform(x_train[["clubs"]])
x_test[["clubs"]] = imputer.transform(x_test[["clubs"]])
print(x_train.isna().sum(),x_test.isna().sum())

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
en_train = encoder.fit_transform(x_train[["course"]])
en_test = encoder.transform(x_test[["course"]])
names = encoder.get_feature_names_out(["course"])
en_train = pd.DataFrame(en_train,columns=names, index=train_idx)
en_test = pd.DataFrame(en_test,columns=names, index=test_idx)
x_train = pd.concat([x_train.drop(columns=["course"]), en_train], axis=1)
x_test = pd.concat([x_test.drop(columns=["course"]), en_test], axis=1)
print(x_train.columns)
print(x_test.columns)
new_courses = pd.DataFrame({
    "course": [encoder.categories_[0][0], "Неизвестный курс"]
})
new_encoded = encoder.transform(new_courses)
print(pd.DataFrame(new_encoded, columns=names, index=new_courses["course"]))
