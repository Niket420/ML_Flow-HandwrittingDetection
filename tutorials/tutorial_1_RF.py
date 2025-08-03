from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Set MLflow config
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("MLflow tutorial_1")

# Parameters
params = {
    "n_estimators": 100,  # ✅ fixed typo
    "max_depth": 3,
    "random_state": 42,
}

# Load data
data = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

with mlflow.start_run():

    clf = RandomForestClassifier(**params)
    clf.fit(x_train, y_train)


    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", acc)


    mlflow.log_params(params)
    mlflow.log_metric("accuracy", acc)
    mlflow.set_tag("Training Info", "Basic RF model for Iris dataset")

    signature = infer_signature(x_train, clf.predict(x_train))
    mlflow.sklearn.log_model(
        sk_model=clf,
        artifact_path="iris_model",
        signature=signature,
        input_example=x_train,
        registered_model_name="tracking-tutorial_1",
    )