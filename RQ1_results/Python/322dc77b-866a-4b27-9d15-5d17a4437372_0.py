from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Define the pipeline
pipeline = Pipeline([
    ('scaling', StandardScaler()),      # Preprocessing step
    ('svm', SVC()),                     # Estimator 1
    ('random_forest', RandomForestClassifier())   # Estimator 2
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Make predictions using the pipeline
y_pred = pipeline.predict(X_test)
