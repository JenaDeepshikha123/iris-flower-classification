import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Page configuration
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)


# Title
st.title("🌸 Iris Flower Classification")
st.write("### Machine Learning Classification using Multiple Models")
st.write(
    "This application predicts the species of an Iris flower "
    "using Logistic Regression, KNN, and Naive Bayes."
)


# Load dataset
df = pd.read_csv("iris.csv")

# Remove Id column
if "Id" in df.columns:
    df = df.drop(columns=["Id"])


# Features and target
X = df[
    ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
]
y = df["Species"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create models
lr = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=5)
nb = GaussianNB()


# Train models
lr.fit(X_train, y_train)
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)


# Sidebar
st.sidebar.header("🌸 Enter Flower Measurements")

sepal_length = st.sidebar.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.sidebar.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.sidebar.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.sidebar.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)


# Input data
input_data = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]],
    columns=X.columns
)


# Select model
st.subheader("🤖 Select Machine Learning Model")

model_name = st.selectbox(
    "Choose a model",
    [
        "Logistic Regression",
        "KNN",
        "Naive Bayes"
    ]
)


# Prediction
if st.button("🔮 Predict Iris Flower"):

    if model_name == "Logistic Regression":
        prediction = lr.predict(input_data)[0]

    elif model_name == "KNN":
        prediction = knn.predict(input_data)[0]

    else:
        prediction = nb.predict(input_data)[0]

    st.success(f"🌸 Predicted Species: {prediction}")


# Compare models
st.subheader("📊 Compare All Models")

if st.button("Compare Predictions"):

    predictions = {
        "Model": [
            "Logistic Regression",
            "KNN",
            "Naive Bayes"
        ],
        "Prediction": [
            lr.predict(input_data)[0],
            knn.predict(input_data)[0],
            nb.predict(input_data)[0]
        ]
    }

    results = pd.DataFrame(predictions)

    st.table(results)


# Dataset
with st.expander("📋 View Dataset"):
    st.dataframe(df)


# Accuracy
with st.expander("📈 Model Accuracy"):

    lr_accuracy = accuracy_score(
        y_test,
        lr.predict(X_test)
    )

    knn_accuracy = accuracy_score(
        y_test,
        knn.predict(X_test)
    )

    nb_accuracy = accuracy_score(
        y_test,
        nb.predict(X_test)
    )

    accuracy_df = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "KNN",
            "Naive Bayes"
        ],
        "Accuracy": [
            round(lr_accuracy * 100, 2),
            round(knn_accuracy * 100, 2),
            round(nb_accuracy * 100, 2)
        ]
    })

    st.dataframe(accuracy_df)