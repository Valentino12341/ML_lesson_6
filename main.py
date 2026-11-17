#Random Forest 
#-------------
# why    faster program   more  correct outcomes 


import pandas as pd 


data = pd.read_csv("studentmat.csv")
print(data.head())
print(data.info())

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

data["school"] = encoder.fit_transform(data["school"])
data["sex"] = encoder.fit_transform(data["sex"])
data["famsize"] = encoder.fit_transform(data["famsize"])
data["Mjob"] = encoder.fit_transform(data["Mjob"])
data["Fjob"] = encoder.fit_transform(data["Fjob"])
data["reason"] = encoder.fit_transform(data["reason"])
data["guardian"] = encoder.fit_transform(data["guardian"])
data["schoolsup"] = encoder.fit_transform(data["schoolsup"])
data["famsup"] = encoder.fit_transform(data["famsup"])
data["paid"] = encoder.fit_transform(data["paid"])
data["activities"] = encoder.fit_transform(data["activities"])
data["nursery"] = encoder.fit_transform(data["nursery"])
data["higher"] = encoder.fit_transform(data["higher"])
data["internet"] = encoder.fit_transform(data["internet"])
data["romantic"] = encoder.fit_transform(data["romantic"])
data["Pstatus"] = encoder.fit_transform(data["Pstatus"])

x = data[["school", "sex", "age", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]]
y = data["G3"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.05,random_state=2)


from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100)

classifier.fit(x_train,y_train)
prediction = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix,accuracy_score

accuracy = accuracy_score(y_test,prediction)

accuracy = round(accuracy*100,2)

print("this thingy is ",accuracy,"procent acurate ")

import seaborn as sns
import matplotlib.pyplot as plt

matrix = confusion_matrix(y_test,prediction)
sns.heatmap(matrix,annot= True,fmt="d")
plt.show()
plt.title("title")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

print
