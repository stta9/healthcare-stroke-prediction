# Healthcare: Stroke prediction

### Installation
1. Clone the repository:
```bash
git clone https://github.com/stta9/healthcare-stroke-prediction.git
cd healthcare-stroke-prediction
```
2. Install the required packages:
```bash
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install imbalanced-learn
```

### Dataset

The dataset used for this project is the Healthcare Stroke Dataset. It contains various features related to patients' health metrics and the target variable stroke, indicating whether the patient had a stroke.

#### Data Preprocessing

Load the dataset:
```python
df = pd.read_csv("/content/healthcare-dataset-stroke-data.csv.xls")
```
Remove unnecessary columns and duplicates:
```python
df.drop(["id"], axis=1, inplace=True)
df.drop_duplicates(inplace=True)
```
Handle missing values:

```python
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
```
Remove irrelevant entries:
```python
df.drop(df[df['gender'] == "Other"].index, axis=0, inplace=True)
```
Encode categorical variables:
```python
label_encoder = LabelEncoder()
for col in categorical:
    df[col] = label_encoder.fit_transform(df[col])
```
#### Exploratory Data Analysis
```python
df.info()
```
Visualize numerical features:
```python
plt.figure(figsize=(15, 5))
for i in range(len(numerical)):
    plt.subplot(1, len(numerical), i + 1)
    sns.boxplot(x=df[numerical[i]])
plt.show()
```
Visualize categorical features:
```python
figure, axis = plt.subplots(4, 2, figsize=(10, 15))
plt.subplots_adjust(hspace=0.5, wspace=0.3)
for i, column_name in enumerate(categorical + ['stroke']):
    row = i // 2
    col = i % 2
    barplot = sns.countplot(ax=axis[row, col], x=df[column_name], hue=df[column_name])
    barplot.set_xticklabels(barplot.get_xticklabels(), rotation=30, size=8)
plt.show()
```
#### Modeling

Split the data into training and testing sets:
```python
X = df.drop(columns=['stroke'])
y = df['stroke']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```
#### Train and evaluate models:

#### Decision Tree Classifier:

```python
tree_model = DecisionTreeClassifier(criterion='entropy')
tree_model.fit(X_train, y_train)
y_pred = tree_model.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
```

#### Logistic Regression:
```python
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
```

#### Random Forest Classifier:
```python
rf_model = RandomForestClassifier(n_estimators=150, criterion='entropy', random_state=123)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
```

## Conclusion
Based on the accuracy results, the Logistic Regression model performed the best among the three models tested.

