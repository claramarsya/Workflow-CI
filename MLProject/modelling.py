import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, f1_score

def train_model(data_path):
    df = pd.read_csv(data_path)
    
    text_col = "clean_title"
    label_col = "real"
    
    if text_col not in df.columns:
        raise Exception(f"Kolom '{text_col}' tidak ditemukan! Apakah file hasil preprocessing benar?")
    
    df = df.dropna(subset=[text_col])
    
    df[text_col] = df[text_col].astype(str)
    
    df[text_col] = df[text_col].replace("", "empty")
    
    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col],
        df[label_col],
        test_size=0.2,
        random_state=42,
        stratify=df[label_col]
    )

    mlflow.set_experiment("FakeNews_Clara")

    with mlflow.start_run():
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("classifier", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(
                max_iter=200, 
                class_weight='balanced',
                solver='liblinear'
            ))
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_weighted", f1_weighted)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="FakeNewsNet_preprocessing.csv",
        help="Path ke dataset hasil preprocessing"
    )
    args = parser.parse_args()

    train_model(args.data_path)