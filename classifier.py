import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import itertools

# confusion matrix  plot: source Matplotlib document
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# read cleaned data
data = pd.read_csv("cleaned_data.csv")
data = data.drop(["Unnamed: 0", "application_type", "zip_code", "addr_state", "earliest_cr_line",
                  "issue_d", "id"], axis=1)
data.info(verbose=True, null_counts=True)

# create one hot labels
labels = pd.get_dummies(data=data["loan_status"])["Fully Paid"]
data = data.drop("loan_status", axis=1)

# convert categorical variables to numeric
list_cat = ["term", "grade", "sub_grade", "emp_title", "emp_length", "home_ownership", "verification_status",
            "purpose", "initial_list_status"]
le = LabelEncoder()
for i in list_cat:
    print("Encoding labels ---")
    le.fit(data[i])
    data[i] = le.transform(data[i])

# Normalizing the data and then training classifier
x = data.values
scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(x))

# train test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=42)


def trainAndTest(X_train, X_test, y_train, y_test, clf_name="RF"):
    if clf_name=="RF":
        classifier = RandomForestClassifier()
    elif clf_name=="LR":
        classifier = LogisticRegression(penalty='l1', C=0.01)
    elif clf_name=="NB":
        classifier = BernoulliNB()
    elif clf_name=="SVM":
        classifier = SVC(kernel="linear")
    classifier.fit(X=X_train, y=y_train)
    predictions = classifier.predict(X=X_test)
    print("Classification accuracy for %s :" % clf_name, accuracy_score(y_test, predictions))
    cm = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(confusion_matrix(y_test, predictions),
                          ["charged_off", "paid"])


trainAndTest(X_train, X_test, y_train, y_test, clf_name="RF")
trainAndTest(X_train, X_test, y_train, y_test, clf_name="LR")
trainAndTest(X_train, X_test, y_train, y_test, clf_name="NB")

# for svm use limited data
sampled_data = data.sample(n=50000, random_state=42)
sampled_labels = labels.sample(n=50000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(sampled_data, sampled_labels, test_size=0.33, random_state=42)
trainAndTest(X_train, X_test, y_train, y_test, clf_name="SVM")


