import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from imblearn.under_sampling import RandomUnderSampler

df = pd.read_csv("Creditcard_data.csv")

print("Original class distribution:")
print(df['Class'].value_counts())


X = df.drop('Class', axis=1)
y = df['Class']

rus = RandomUnderSampler(random_state=42)
X_balanced, y_balanced = rus.fit_resample(X, y)

X_balanced = X_balanced.reset_index(drop=True)
y_balanced = y_balanced.reset_index(drop=True)

print("\nBalanced class distribution:")
print(y_balanced.value_counts())


samples = {}


# Sampling1 – Simple Random Sampling
X_train1, X_test1, y_train1, y_test1 = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=1
)
samples['Sampling1'] = (X_train1, X_test1, y_train1, y_test1)


# Sampling2 – Systematic Sampling
indices = np.arange(0, len(X_balanced), 2)

X_train2 = X_balanced.iloc[indices]
y_train2 = y_balanced.iloc[indices]

X_test2 = X_balanced.drop(indices).reset_index(drop=True)
y_test2 = y_balanced.drop(indices).reset_index(drop=True)

samples['Sampling2'] = (X_train2, X_test2, y_train2, y_test2)


# Sampling3 – Stratified Sampling
X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X_balanced,
    y_balanced,
    test_size=0.3,
    stratify=y_balanced,
    random_state=2
)
samples['Sampling3'] = (X_train3, X_test3, y_train3, y_test3)


# Sampling4 – Bootstrap Sampling
boot_indices = np.random.choice(
    len(X_balanced),
    size=len(X_balanced),
    replace=True
)

X_train4 = X_balanced.iloc[boot_indices]
y_train4 = y_balanced.iloc[boot_indices]

X_test4 = X_balanced.drop(boot_indices).reset_index(drop=True)
y_test4 = y_balanced.drop(boot_indices).reset_index(drop=True)

samples['Sampling4'] = (X_train4, X_test4, y_train4, y_test4)


# Sampling5 – K-Fold (Cross Validation Sampling)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=3)
train_idx, test_idx = next(skf.split(X_balanced, y_balanced))

X_train5 = X_balanced.iloc[train_idx]
y_train5 = y_balanced.iloc[train_idx]
X_test5 = X_balanced.iloc[test_idx]
y_test5 = y_balanced.iloc[test_idx]

samples['Sampling5'] = (X_train5, X_test5, y_train5, y_test5)


models = {
    'M1_LogisticRegression': LogisticRegression(max_iter=1000),
    'M2_DecisionTree': DecisionTreeClassifier(),
    'M3_RandomForest': RandomForestClassifier(n_estimators=50),
    'M4_SVM': SVC(),
    'M5_KNN': KNeighborsClassifier()
}


results = pd.DataFrame(
    index=models.keys(),
    columns=samples.keys()
)

for model_name, model in models.items():
    for sample_name, (X_tr, X_te, y_tr, y_te) in samples.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred) * 100
        results.loc[model_name, sample_name] = round(acc, 2)



print("\nAccuracy Table:")
print(results)


print("\nBest Sampling Technique per Model:")
for model in results.index:
    best_sampling = results.loc[model].astype(float).idxmax()
    best_accuracy = results.loc[model].astype(float).max()
    print(f"{model} → {best_sampling} ({best_accuracy}%)")

results.index = ['M1', 'M2', 'M3', 'M4', 'M5']
results = results.astype(float)

results.to_csv("sampling_results.csv")

print("\nCSV file 'sampling_results.csv' has been created successfully!")
