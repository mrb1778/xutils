import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score


class AutoMLClassifier:
    def __init__(self, scoring_function='balanced_accuracy', n_iter=50):
        self.scoring_function = scoring_function
        self.n_iter = n_iter

    def fit(self, X, y):
        X_train = X
        y_train = y

        categorical_values = []

        cat_subset = X_train.select_dtypes(include=['object', 'category', 'bool'])

        for i in range(cat_subset.shape[1]):
            categorical_values.append(list(cat_subset.iloc[:, i].dropna().unique()))

        num_pipeline = Pipeline([
            ('cleaner', SimpleImputer()),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ('cleaner', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(sparse=False, categories=categorical_values))
        ])

        preprocessor = ColumnTransformer([
            ('numerical', num_pipeline, make_column_selector(dtype_exclude=['object', 'category', 'bool'])),
            ('categorical', cat_pipeline, make_column_selector(dtype_include=['object', 'category', 'bool']))
        ])

        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('feature_selector', SelectKBest(f_classif, k='all')),
            ('estimator', LogisticRegression())])

        total_features = preprocessor.fit_transform(X_train).shape[1]

        optimization_grid = [
            {
                'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [LogisticRegression()]},
            {
                'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [KNeighborsClassifier()],
                'estimator__weights': ['uniform', 'distance'],
                'estimator__n_neighbors': np.arange(1, 20, 1)
            },
            {
                'preprocessor__numerical__scaler': [None],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [RandomForestClassifier(random_state=0)],
                'estimator__n_estimators': np.arange(5, 500, 10),
                'estimator__criterion': ['gini', 'entropy']
            },
            {
                'preprocessor__numerical__scaler': [None],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [GradientBoostingClassifier(random_state=0)],
                'estimator__n_estimators': np.arange(5, 500, 10),
                'estimator__learning_rate': np.linspace(0.1, 0.9, 20),
            },
            {
                'preprocessor__numerical__scaler': [None],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [DecisionTreeClassifier(random_state=0)],
                'estimator__criterion': ['gini', 'entropy']
            },
            {
                'preprocessor__numerical__scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
                'preprocessor__numerical__cleaner__strategy': ['mean', 'median'],
                'feature_selector__k': list(np.arange(1, total_features, 5)) + ['all'],
                'estimator': [LinearSVC(random_state=0)],
                'estimator__C': np.arange(0.1, 1, 0.1),

            }
        ]

        # Logistic regression

        # K-nearest neighbors

        # Random Forest

        # Gradient boosting

        # Decision tree

        # Linear SVM

        search = RandomizedSearchCV(
            model_pipeline,
            optimization_grid,
            n_iter=self.n_iter,
            scoring=self.scoring_function,
            n_jobs=-1,
            random_state=0,
            verbose=3,
            cv=5
        )

        search.fit(X_train, y_train)
        self.best_estimator_ = search.best_estimator_
        self.best_pipeline = search.best_params_

    def predict(self, X, y=None):
        return self.best_estimator_.predict(X)

    def predict_proba(self, X, y=None):
        return self.best_estimator_.predict_proba(X)


def test_automl():
    d = load_breast_cancer()
    y = d['target']
    X = pd.DataFrame(d['data'], columns=d['feature_names'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = AutoMLClassifier()
    model.fit(X_train, y_train)
    
    print(balanced_accuracy_score(y_test, model.predict(X_test)))
    print(model.best_pipeline)
