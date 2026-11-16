import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, log_loss, mean_absolute_error, mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_sample_weight

# ===========================================================================
# Notebook code cell 2: Step 0 — Setup
# ===========================================================================
import warnings
warnings.filterwarnings("ignore")   # keep output focused on results, not library version notices

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, log_loss,
    mean_absolute_error, mean_squared_error, r2_score
)

pd.set_option("display.max_columns", None)
plt.rcParams["figure.figsize"] = (6, 4)
print("All libraries loaded")


# ===========================================================================
# Notebook code cell 5: 1. Load the dataset
# ===========================================================================
drug = pd.read_csv("data/drug200.csv")
drug.head()


# ===========================================================================
# Notebook code cell 7: 2. Understand the data
# ===========================================================================
drug.info()


# ===========================================================================
# Notebook code cell 8: 2. Understand the data
# ===========================================================================
drug.describe()


# ===========================================================================
# Notebook code cell 9: 2. Understand the data
# ===========================================================================
drug.columns


# ===========================================================================
# Notebook code cell 11: 3. Clean the data
# ===========================================================================
print("Missing values per column:\n", drug.isnull().sum())
print("\nDuplicate rows:", drug.duplicated().sum())
print("\nData types:\n", drug.dtypes)


# ===========================================================================
# Notebook code cell 14: 4. Explore the data
# ===========================================================================
sns.countplot(x="Drug", data=drug)
plt.title("Target variable distribution — Drug")
plt.show()
drug["Drug"].value_counts()


# ===========================================================================
# Notebook code cell 16: 4. Explore the data
# ===========================================================================
sns.histplot(drug["Na_to_K"], kde=True)
plt.title("Distribution of Na_to_K")
plt.show()


# ===========================================================================
# Notebook code cell 18: 5. Preprocess the data
# ===========================================================================
drug_enc = drug.copy()

le_sex = LabelEncoder()
drug_enc["Sex"] = le_sex.fit_transform(drug_enc["Sex"])          # F=0, M=1

# BP and Cholesterol have a natural order, so we map them by hand instead of LabelEncoder
drug_enc["BP"] = drug_enc["BP"].map({"LOW": 0, "NORMAL": 1, "HIGH": 2})
drug_enc["Cholesterol"] = drug_enc["Cholesterol"].map({"NORMAL": 0, "HIGH": 1})

drug_enc.head()


# ===========================================================================
# Notebook code cell 20: 6. Define X and y
# ===========================================================================
X = drug_enc.drop("Drug", axis=1)
y = drug_enc["Drug"]
print(X.shape, y.shape)


# ===========================================================================
# Notebook code cell 22: 7. Train-test split
# ===========================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train:", X_train.shape, " Test:", X_test.shape)


# ===========================================================================
# Notebook code cell 24: 8–10. Select algorithm → Train → Predict
# ===========================================================================
# Decision Tree
drug_tree = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
drug_tree.fit(X_train, y_train)
pred_tree = drug_tree.predict(X_test)

# Logistic Regression — One vs Rest (explicit)
log_ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
log_ovr.fit(X_train, y_train)
pred_log_ovr = log_ovr.predict(X_test)

# Logistic Regression — One vs One (explicit, for comparison)
log_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
log_ovo.fit(X_train, y_train)
pred_log_ovo = log_ovo.predict(X_test)

# SVM (sklearn's SVC is One-vs-One by default for multi-class)
svm_model = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
svm_model.fit(X_train, y_train)
pred_svm = svm_model.predict(X_test)

print("All 4 models trained and predicted")


# ===========================================================================
# Notebook code cell 26: 11. Evaluate — classification metrics
# ===========================================================================
def evaluate(name, y_true, y_pred, proba=None):
    print(f"--- {name} ---")
    print("Accuracy :", round(accuracy_score(y_true, y_pred), 3))
    print("Precision:", round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 3))
    print("Recall   :", round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 3))
    print("F1 score :", round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 3))
    if proba is not None:
        n_classes = len(np.unique(y_true))
        if n_classes == 2 and proba.ndim == 2 and proba.shape[1] == 2:
            auc = roc_auc_score(y_true, proba[:, 1])          # binary: use probability of the positive class only
        else:
            auc = roc_auc_score(y_true, proba, multi_class="ovr")  # multi-class: full probability matrix
        print("ROC-AUC  :", round(auc, 3))
    print()

evaluate("Decision Tree", y_test, pred_tree, drug_tree.predict_proba(X_test))
evaluate("Logistic Regression (OvR)", y_test, pred_log_ovr, log_ovr.predict_proba(X_test))
evaluate("Logistic Regression (OvO)", y_test, pred_log_ovo)   # OvO wrapper has no predict_proba
evaluate("SVM (rbf) — unscaled features", y_test, pred_svm, svm_model.predict_proba(X_test))


# ===========================================================================
# Notebook code cell 28: 11. Evaluate — classification metrics
# ===========================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)          # fit on train only, transform both — never fit on test data

svm_scaled = SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
svm_scaled.fit(X_train_scaled, y_train)
pred_svm_scaled = svm_scaled.predict(X_test_scaled)

evaluate("SVM (rbf) — scaled features", y_test, pred_svm_scaled, svm_scaled.predict_proba(X_test_scaled))


# ===========================================================================
# Notebook code cell 30: 11. Evaluate — classification metrics
# ===========================================================================
print("Confusion matrix — Decision Tree")
print(confusion_matrix(y_test, pred_tree))
print()
print(classification_report(y_test, pred_tree, zero_division=0))


# ===========================================================================
# Notebook code cell 32: 12. Compare models
# ===========================================================================
comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Logistic Regression (OvR)", "Logistic Regression (OvO)", "SVM (rbf, scaled)"],
    "Accuracy": [
        accuracy_score(y_test, pred_tree),
        accuracy_score(y_test, pred_log_ovr),
        accuracy_score(y_test, pred_log_ovo),
        accuracy_score(y_test, pred_svm_scaled),
    ],
    "F1 (weighted)": [
        f1_score(y_test, pred_tree, average="weighted", zero_division=0),
        f1_score(y_test, pred_log_ovr, average="weighted", zero_division=0),
        f1_score(y_test, pred_log_ovo, average="weighted", zero_division=0),
        f1_score(y_test, pred_svm_scaled, average="weighted", zero_division=0),
    ],
}).sort_values("F1 (weighted)", ascending=False).reset_index(drop=True)
comparison


# ===========================================================================
# Notebook code cell 34: 13. Interpret results
# ===========================================================================
importances = pd.Series(drug_tree.feature_importances_, index=X.columns).sort_values(ascending=False)
importances


# ===========================================================================
# Notebook code cell 35: 13. Interpret results
# ===========================================================================
plt.figure(figsize=(12, 6))
plot_tree(drug_tree, feature_names=X.columns, class_names=drug_tree.classes_, filled=True, fontsize=8)
plt.title("Decision Tree — drug200")
plt.show()


# ===========================================================================
# Notebook code cell 38: Part A2 — Handling imbalanced classes
# ===========================================================================
# Fix 1: class_weight='balanced' — tells the model to pay more attention to rare classes automatically
tree_balanced = DecisionTreeClassifier(criterion="entropy", max_depth=4, class_weight="balanced", random_state=42)
tree_balanced.fit(X_train, y_train)
pred_balanced = tree_balanced.predict(X_test)
evaluate("Decision Tree (class_weight=balanced)", y_test, pred_balanced, tree_balanced.predict_proba(X_test))


# ===========================================================================
# Notebook code cell 39: Part A2 — Handling imbalanced classes
# ===========================================================================
# Fix 2: compute_sample_weight — same idea, but you compute the weights yourself and pass them into .fit()
sample_weights = compute_sample_weight("balanced", y_train)
tree_weighted = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=42)
tree_weighted.fit(X_train, y_train, sample_weight=sample_weights)
pred_weighted = tree_weighted.predict(X_test)
evaluate("Decision Tree (sample_weight)", y_test, pred_weighted, tree_weighted.predict_proba(X_test))


# ===========================================================================
# Notebook code cell 43: 1–2. Load and understand
# ===========================================================================
fuel = pd.read_csv("data/FuelConsumptionCo2.csv")
print(fuel.shape)
fuel[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]].describe()


# ===========================================================================
# Notebook code cell 45: 3. Clean
# ===========================================================================
print(fuel.isnull().sum().sum(), "missing values")
print(fuel.duplicated().sum(), "duplicate rows")


# ===========================================================================
# Notebook code cell 47: 4. Explore — correlation with the target
# ===========================================================================
fuel_num = fuel[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"]]
sns.heatmap(fuel_num.corr(), annot=True, cmap="Blues")
plt.title("Correlation with CO2EMISSIONS")
plt.show()


# ===========================================================================
# Notebook code cell 50: 5–7. Preprocess, define X/y, split
# ===========================================================================
X_fuel = fuel[["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]]
y_fuel = fuel["CO2EMISSIONS"]

Xf_train, Xf_test, yf_train, yf_test = train_test_split(X_fuel, y_fuel, test_size=0.2, random_state=42)
print(Xf_train.shape, Xf_test.shape)


# ===========================================================================
# Notebook code cell 52: 8–10. Select, train, predict — Linear Regression & Decision Tree Regressor
# ===========================================================================
lin_reg = LinearRegression()
lin_reg.fit(Xf_train, yf_train)
pred_lin = lin_reg.predict(Xf_test)

tree_reg = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_reg.fit(Xf_train, yf_train)
pred_tree_reg = tree_reg.predict(Xf_test)

print("Both regression models trained")


# ===========================================================================
# Notebook code cell 54: 11. Evaluate — regression metrics
# ===========================================================================
def evaluate_reg(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"--- {name} ---")
    print("MAE :", round(mae, 2))
    print("MSE :", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2  :", round(r2, 3))
    print()
    return mae, mse, rmse, r2

lin_scores = evaluate_reg("Linear Regression", yf_test, pred_lin)
tree_scores = evaluate_reg("Decision Tree Regressor", yf_test, pred_tree_reg)


# ===========================================================================
# Notebook code cell 56: 12. Compare models
# ===========================================================================
reg_comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree Regressor"],
    "MAE": [lin_scores[0], tree_scores[0]],
    "RMSE": [lin_scores[2], tree_scores[2]],
    "R2": [lin_scores[3], tree_scores[3]],
}).sort_values("R2", ascending=False).reset_index(drop=True)
reg_comparison


# ===========================================================================
# Notebook code cell 58: 13. Interpret
# ===========================================================================
coefs = pd.Series(lin_reg.coef_, index=X_fuel.columns)
print("Intercept:", round(lin_reg.intercept_, 2))
coefs


# ===========================================================================
# Notebook code cell 60: Part C — KNN Classification + choosing the best k
# ===========================================================================
tele = pd.read_csv("data/teleCust1000t.csv")
tele["custcat"].value_counts()


# ===========================================================================
# Notebook code cell 61: Part C — KNN Classification + choosing the best k
# ===========================================================================
X_tele = tele.drop("custcat", axis=1)
y_tele = tele["custcat"]

# KNN is distance-based, so scaling is essential (unlike trees)
X_tele_scaled = StandardScaler().fit_transform(X_tele)

Xt_train, Xt_test, yt_train, yt_test = train_test_split(
    X_tele_scaled, y_tele, test_size=0.2, random_state=42
)


# ===========================================================================
# Notebook code cell 62: Part C — KNN Classification + choosing the best k
# ===========================================================================
ks = range(1, 11)
accuracies = []

for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(Xt_train, yt_train)
    acc = accuracy_score(yt_test, knn.predict(Xt_test))
    accuracies.append(acc)

plt.plot(list(ks), accuracies, marker="o")
plt.xlabel("k (number of neighbours)")
plt.ylabel("Test accuracy")
plt.title("Choosing the best k")
plt.show()

best_k = list(ks)[int(np.argmax(accuracies))]
print(f"Best k = {best_k}, accuracy = {max(accuracies):.3f}")


# ===========================================================================
# Notebook code cell 64: Part D — K-Means Clustering (unsupervised bonus)
# ===========================================================================
cluster_data = drug_enc[["Age", "Na_to_K"]]
cluster_scaled = StandardScaler().fit_transform(cluster_data)

wcss = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    km.fit(cluster_scaled)
    wcss.append(km.inertia_)

plt.plot(range(1, 9), wcss, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow method")
plt.show()


# ===========================================================================
# Notebook code cell 65: Part D — K-Means Clustering (unsupervised bonus)
# ===========================================================================
# Based on the elbow, k=3 is a reasonable choice here
kmeans_final = KMeans(n_clusters=3, init="k-means++", random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(cluster_scaled)

print("Cluster centres (scaled space):\n", kmeans_final.cluster_centers_)

plt.scatter(cluster_data["Age"], cluster_data["Na_to_K"], c=clusters, cmap="viridis")
plt.xlabel("Age")
plt.ylabel("Na_to_K")
plt.title("K-Means clusters — drug200 (Age vs Na_to_K)")
plt.show()


# ===========================================================================
# Notebook code cell 67: Part E — Find-S & Candidate Elimination
# ===========================================================================
sport = pd.read_csv("data/ENJOYSPORT.csv")
sport


# ===========================================================================
# Notebook code cell 68: Part E — Find-S & Candidate Elimination
# ===========================================================================
def find_s(data):
    attributes = data.columns[:-1]         # all columns except the target
    target = data.columns[-1]

    # start with the most specific hypothesis: nothing matches
    hypothesis = ["0"] * len(attributes)

    for i, row in data.iterrows():
        if row[target] == 1:                # only look at positive examples
            for j, attr in enumerate(attributes):
                if hypothesis[j] == "0":
                    hypothesis[j] = row[attr]        # first positive example: copy it exactly
                elif hypothesis[j] != row[attr]:
                    hypothesis[j] = "?"              # disagreement: generalise to "don't care"
    return hypothesis

final_hypothesis = find_s(sport)
print("Attributes:", list(sport.columns[:-1]))
print("Find-S hypothesis:", final_hypothesis)


# ===========================================================================
# Notebook code cell 71: Part F — Logistic Regression + Log Loss
# ===========================================================================
churn = pd.read_excel("data/ChurnData.xlsx")
churn.columns.tolist()[:10]


# ===========================================================================
# Notebook code cell 72: Part F — Logistic Regression + Log Loss
# ===========================================================================
churn_df = churn[["tenure", "age", "address", "income", "ed", "employ", "equip", "churn"]]
churn_df["churn"] = churn_df["churn"].astype(int)

X_churn = churn_df.drop("churn", axis=1)
y_churn = churn_df["churn"]

X_churn_norm = StandardScaler().fit_transform(X_churn)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_churn_norm, y_churn, test_size=0.2, random_state=4
)


# ===========================================================================
# Notebook code cell 73: Part F — Logistic Regression + Log Loss
# ===========================================================================
log_model = LogisticRegression()
log_model.fit(Xc_train, yc_train)

yhat = log_model.predict(Xc_test)
yhat_proba = log_model.predict_proba(Xc_test)

evaluate("Logistic Regression — Churn", yc_test, yhat, yhat_proba)
print("Log loss:", round(log_loss(yc_test, yhat_proba), 3))


# ===========================================================================
# Notebook code cell 75: Part F — Logistic Regression + Log Loss
# ===========================================================================
coeffs = pd.Series(log_model.coef_[0], index=X_churn.columns).sort_values(ascending=False)
coeffs


# ===========================================================================
# Notebook code cell 79: Part H — Smart guesses: what else could show up
# ===========================================================================
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42, class_weight="balanced")
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
evaluate("Random Forest", y_test, pred_rf, rf.predict_proba(X_test))


# ===========================================================================
# Notebook code cell 81: 2. Hyperparameter tuning (natural next step after "choosing the best k")
# ===========================================================================
from sklearn.model_selection import GridSearchCV

param_grid = {"max_depth": [2, 3, 4, 5, 6], "criterion": ["entropy", "gini"]}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring="f1_weighted")
grid.fit(X_train, y_train)

print("Best params:", grid.best_params_)
print("Best CV F1 score:", round(grid.best_score_, 3))


# SECOND SOURCE: MLM_Code_Only PDF — programs appended in source order


# ===========================================================================
# PDF code: 1. Sentiment Analysis: Naive Bayes vs Logistic Regression (Text)
# ===========================================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = {
    'text': [
         "I absolutely love this product, it works perfectly!",
         "This is the worst purchase I have ever made.",
         "It's okay, nothing special but does the job",
         "Fantastic customer service and beautiful design",
         "Horrible quality, it broke on the first day",
         "I am highly satisfied with the fast shipping",
         "Waste of money. do not but this product",
         "Pretty good value for the price, I recommend it",
         "item arrived late and the product was damaged, very disappointed"
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative',
                   'positive', 'negative', 'positive', 'negative']
}

# load into a pandas dataframe
df = pd.DataFrame(data)

# 1. split dataset into features (X) and target labels (Y)
x = df['text']
y = df['sentiment']

# 2. split into training and testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. text vectorization (TF-IDF feature extraction)
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# 4. initialize and train classifiers
lr_model = LogisticRegression()
lr_model.fit(x_train_vec, y_train)

nb_model = MultinomialNB()
nb_model.fit(x_train_vec, y_train)

# 5. evaluate and compare models
lr_preds = lr_model.predict(x_test_vec)
nb_preds = nb_model.predict(x_test_vec)

print("Model Accuracy Comparison")
print(f"Logistic Regression Accuracy:{accuracy_score(y_test, lr_preds)*100:.2f}%")
print(f"Multinomial Naive Bayes Accuracy:{accuracy_score(y_test, nb_preds)*100:.2f}%\n")

# 6. predict sentiment on new custom texts
def predict_sentiment(new_texts):
    new_texts_vec = vectorizer.transform(new_texts)

    lr_predictions = lr_model.predict(new_texts_vec)
    nb_predictions = nb_model.predict(new_texts_vec)

    print("testing custom input")
    for text, lr_pred, nb_pred in zip(new_texts, lr_predictions, nb_predictions):
        print(f"Text: '{text}'")
        print(f"LR Prediction:**{lr_pred.upper()}**")
        print(f"NB Prediction:**{nb_pred.upper()}**\n")

custom_samples = ["The item arrived broken and late",
                  "it wokrs incredibly well, love it!",
                  "looks good"]
predict_sentiment(custom_samples)

from sklearn.datasets import fetch_20newsgroups


# ===========================================================================
# PDF code: 2. Multinomial Naive Bayes on the 20 Newsgroups Dataset
# ===========================================================================
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# load the 20 newsgroup dataset
newsgroups = fetch_20newsgroups(subset='all')
x, y = newsgroups.data, newsgroups.target

# split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# initialize the TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.5)

# fit and transform the training data and transform the test data
x_train_tfidf = vectorizer.fit_transform(x_train)
x_test_tfidf = vectorizer.transform(x_test)

# initialize the multinomial naive bayes classifier
nb_classifier = MultinomialNB()

# train the classifier on the training data
nb_classifier.fit(x_train_tfidf, y_train)

# predict the labels for the test data
y_pred = nb_classifier.predict(x_test_tfidf)

# calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

# generate a classification report
class_report = classification_report(y_test, y_pred, target_names=newsgroups.target_names)

print(f"Accuracy:{accuracy}\n")
print(f"Classification Report:\n{class_report}")


# ===========================================================================
# PDF code: 3. Gaussian Naive Bayes on the Iris Dataset (+ Credit Card Data Load)
# ===========================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report

# Load the data set
iris = load_iris()
x = iris.data
y = iris.target

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. Initialize and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Make predictions on the test set
y_pred = model.predict(X_test)

# 5. Evaluate the model's performance
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import pandas as pd
raw_data = pd.read_csv("D:\\MSC BA\\SEM 3\\ClassWork\\MLM\\creditcard(in).csv")
raw_data

X = raw_data.drop("Class", axis=1)
y = raw_data["Class"]


# ===========================================================================
# PDF code: 4. Pipeline, PCA, KNN, Confusion Matrix and GridSearchCV
# ===========================================================================
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.metrics import confusion_matrix

data = load_iris()
X, y = data.data, data.target
labels = data.target_names

# standard scaler, PCA and KNeighborsClassifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),                # step 1 - standardize the features
    ('pca', PCA(n_components=2)),                # step 2 - reduce the dimensions to 2 using PCA
    ('knn', KNeighborsClassifier(n_neighbors=5)) # step 3 - K-Nearest neighbours classifier
])

# stratify is to be included in pipeline programs
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# fit the pipeline into training set
pipeline.fit(x_train, y_train)

# measure the pipeline accuracy on the test data
test_score = pipeline.score(x_test, y_test)
print(f"{test_score:.3f}")

# get the model predictions
y_pred = pipeline.predict(x_test)

# generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# create a plot for the confusion matrix
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
             xticklabels=labels, yticklabels=labels)

# set the title and labels
plt.title("Classification pipeline confusion matrix")
plt.xlabel('Predicted')
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# make the pipeline without specifying any parameters yet
pipeline = Pipeline(
    [('scaler', StandardScaler()),
     ('pca', PCA()),
     ('knn', KNeighborsClassifier())]
)

# define a model parameter grid to search over
param_grid = {'pca__n_components': [2, 3],
              'knn__n_neighbors': [3, 5, 6]}

# choose a cross validation method
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# determine the best parameters
best_model = GridSearchCV(estimator=pipeline, param_grid=param_grid,
                          cv=cv, scoring='accuracy', verbose=2)

# fit the best gridsearchcv model to the training data
best_model.fit(x_train, y_train)

# evaluate the accuracy of the best model on the test set
test_score = best_model.score(x_test, y_test)
print(f"{test_score:.3f}")
# display the best parameters


# ===========================================================================
# PDF code: Program 4 continued: GridSearchCV best parameters
# ===========================================================================
best_model.best_params_


# ===========================================================================
# PDF code: 5. Random Forest Classifier on Iris
# ===========================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X = data.data
y = data.target

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classsification_report_str = classification_report(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{classsification_report_str}")


# ===========================================================================
# PDF code: 6. XGBoost Classifier on Iris (Feature Importance + Confusion Matrix)
# ===========================================================================
from xgboost import XGBClassifier, plot_importance
import seaborn as sns

X = data.data
y = data.target

# splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

xgb_classifier = XGBClassifier(objective='multi:softprob', num_class=3,
                               eval_metric='mlogloss', random_state=42,
                               use_label_encoder=False)
xgb_classifier.fit(X_train, y_train)

y_pred_xgb = xgb_classifier.predict(X_test)

print(f"XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb)}")
print(f"XGBoost Classification Report:\n{classification_report(y_test, y_pred_xgb)}")
plot_importance(xgb_classifier)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred_xgb)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# ===========================================================================
# PDF code: 7. Churn Prediction: Random Forest, Gradient Boosting, XGBoost with
# ===========================================================================
Cross-Validation
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, ConfusionMatrixDisplay)

from xgboost import XGBClassifier

# Load data
df = pd.read_excel(r"D:\MSC BA\SEM 3\ClassWork\MLM\ChurnData.xlsx")
df.head()

target_col = 'churn'

if target_col not in df.columns:
    target_col = df.columns[-1]
    print(f"'{target_col}' not found. Using the last column '{target_col}' as target")

# Handle Missing Values
df = df.dropna()

# Encode categorical text variables
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# Split features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Split into train/test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

# Feature scaling (optional for tree-based models, required for SVM, Logistic Regression, etc.)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Model Initialization
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42,
                             eval_metric='logloss', use_label_encoder=False)
}

# evaluate models using cross-validation loop
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"Model: {name}")

    # Fit model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'Confusion Matrix for {name}')


# ===========================================================================
# PDF code: Program 7 continued: Cross-validation and feature importance
# ===========================================================================
    plt.show()

    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=kf)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {np.mean(cv_scores)}",
         f"Standard Deviation of CV scores: {np.std(cv_scores)}")
    print("-" * 50)

print(" Top 5 Feature Importance:")
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(min(5, len(X.columns))):
    print(f"{i + 1}. Feature: {X.columns[indices[i]]}, Importance: {importances[indices[i]]:.4f}")
