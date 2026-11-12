import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

df = pd.read_csv("school_progress.csv")
x = df[["grade", "attempts", "average_score", "course", "city", "clubs"]]
y=df["status"]
# print(x.dtypes)
# print(y.dtypes)
# print(x.isna().sum())
# print(y.isna().sum())
# print(x[["course", "city", "clubs"]].dropna().nunique())
#"course", "city", "clubs" в кодировании категорий. clubs avg_score в dropna. все остальые крутые числовые
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(x, y, groups=df["student_id"]))
print(len(set(df["student_id"].iloc[train_idx]) & set(df["student_id"].iloc[test_idx])) > 0)
x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
# imputer = SimpleImputer(strategy="median")
# x_train["average_score"] = imputer.fit_transform(x_train[["average_score"]])
# x_test["average_score"] = imputer.transform(x_test[["average_score"]])
# imputer = SimpleImputer(strategy="constant", fill_value="не указано")
# x_train[["clubs"]] = imputer.fit_transform(x_train[["clubs"]])
# x_test[["clubs"]] = imputer.transform(x_test[["clubs"]])
# print(x_train.isna().sum(),x_test.isna().sum())
#
# okak = x_train["course"].iloc[0]
# encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# en_train = encoder.fit_transform(x_train[["course"]])
# en_test = encoder.transform(x_test[["course"]])
# names = encoder.get_feature_names_out(["course"])
# en_train = pd.DataFrame(en_train,columns=names, index=train_idx)
# en_test = pd.DataFrame(en_test,columns=names, index=test_idx)
# x_train = pd.concat([x_train.drop(columns=["course"]), en_train], axis=1)
# x_test = pd.concat([x_test.drop(columns=["course"]), en_test], axis=1)
# print(x_train.columns)
# print(x_test.columns)
# new_courses = pd.DataFrame({'course': [okak, 'Неизвестный курс']})
# new_encoded = encoder.transform(new_courses)
# print(pd.DataFrame(new_encoded, columns=names))

num_pipeline = Pipeline(steps=[
    ("fill", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
cat_pipeline = Pipeline(steps=[
    ("fill", SimpleImputer(strategy="constant", fill_value="не указано")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])
num_cols = ["grade", "attempts", "average_score"]
cat_cols = ["course", "city", "clubs"]
preprocessor = ColumnTransformer(transformers=[
    ("numbers", num_pipeline, num_cols),
    ("categories", cat_pipeline, cat_cols),
])
model = Pipeline(steps=[
    ("prepare", preprocessor),
    ("tree", DecisionTreeClassifier(max_depth=3, random_state=42))
])
model.fit(x_train,y_train)
prediction = model.predict(x_test)
print(pd.DataFrame(data={"true" : y_test, "predict" : prediction}), accuracy_score(y_test, prediction))
print(x_train.shape)
print(model.named_steps["prepare"].transform(x_train).shape)#клубы курсы и города разбились
preprocessor1 = ColumnTransformer(transformers=[
    ("numbers", num_pipeline, num_cols)
])
model1 = Pipeline(steps=[
    ("prepare", preprocessor1),
    ("tree", DecisionTreeClassifier(max_depth=3, random_state=42))
])
model1.fit(x_train,y_train)
prediction1t = model1.predict(x_test)
prediction1r = model1.predict(x_train)
predictionr = model.predict(x_train)
results = pd.DataFrame({"accuracy":[accuracy_score(y_test,prediction), accuracy_score(y_train,predictionr), accuracy_score(y_test,prediction1t), accuracy_score(y_train,prediction1r)],"model":["model", "model", "model1", "model1"],"type":["test","train","test","train"]})
print(results.groupby(["model","type"])["accuracy"].first())
num_pipeline = Pipeline(steps=[
    ("fill", SimpleImputer(strategy="median"))
])
preprocessor2 = ColumnTransformer(transformers=[
    ("numbers", num_pipeline, num_cols),
    ("categories", cat_pipeline, cat_cols),
])
model2 = Pipeline(steps=[
    ("prepare", preprocessor2),
    ("tree", DecisionTreeClassifier(max_depth=3, random_state=42))
])
model2.fit(x_train,y_train)
prediction2t = model2.predict(x_test)
prediction2r = model2.predict(x_train)
results = pd.DataFrame({
    "accuracy": [
        accuracy_score(y_test, prediction),
        accuracy_score(y_train, predictionr),
        accuracy_score(y_test, prediction1t),
        accuracy_score(y_train, prediction1r),
        accuracy_score(y_test, prediction2t),
        accuracy_score(y_train, prediction2r)
    ],
    "model": ["model", "model", "model1", "model1", "model2", "model2"],
    "type": ["test", "train", "test", "train", "test", "train"]
})
sns.barplot(data=results, x="model", y="accuracy", hue="type")
plt.show()
