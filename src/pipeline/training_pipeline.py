import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                           classification_report, roc_auc_score,
                           precision_recall_curve, average_precision_score)
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import comet_ml
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Dict
from src.components.data_preprocessing import data_preprocessing

from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env
api_key = os.getenv("COMET_API_KEY")

comet_ml.login(api_key)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess the raw data"""
    logger.info("Loading and preprocessing data...")
    df = pd.read_csv(filepath)
    df.set_index("trans_num", inplace=True)
    
    # [Include your complete data_preprocessing() function here]
    # Example placeholder - replace with your actual implementation
    df = data_preprocessing(df)
    
    return df

def encode_and_split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, OneHotEncoder]:
    """Encode features and split into train/validation sets"""
    logger.info("Encoding features...")
    
    # Define columns
    cat_cols = ['time_bucket', 'category', 'amount_bkt', 'population_bkt', 'age_bkt']
    drop_cols = cat_cols + ["trans_date", "cc_num", "gender", "state", "age", 'Unnamed: 0']
    
    # Initialize and fit encoder
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    X_encoded = encoder.fit_transform(df[cat_cols])
    encoded_feat_names = encoder.get_feature_names_out(cat_cols)
    
    # Create encoded DataFrame
    encoded_df = pd.DataFrame(X_encoded, columns=encoded_feat_names, index=df.index)
    data_processed = df.drop(columns=drop_cols)
    final_df = pd.concat([data_processed, encoded_df], axis=1)
    
    # Verification
    logger.info(f"Original shape: {df.shape}")
    logger.info(f"Final shape: {final_df.shape}")
    
    # Split data
    X = final_df.drop(columns="is_fraud")
    y = final_df["is_fraud"]
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val, encoder

def resample_data(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to handle class imbalance"""
    logger.info("Applying SMOTE resampling...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"Resampled training data shape: {X_resampled.shape}")
    return X_resampled, y_resampled

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                experiment) -> Tuple[RandomForestClassifier, Dict]:
    """Train model using HalvingGridSearchCV"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced', None]
    }
    
    base_model = RandomForestClassifier(random_state=42)
    
    search = HalvingGridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        factor=3,
        cv=5,
        aggressive_elimination=True,
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("Starting Halving Grid Search...")
    search.fit(X_train, y_train)
    logger.info("Search completed!")
    
    # Log parameters and results
    experiment.log_parameters(search.best_params_)
    experiment.log_metric("best_cv_score", search.best_score_)
    
    return search.best_estimator_, search.best_params_

def evaluate_model(model: RandomForestClassifier, X_val: pd.DataFrame,
                  y_val: pd.Series, experiment) -> Dict:
    """Evaluate model performance and log metrics"""
    y_pred = model.predict(X_val)
    y_probs = model.predict_proba(X_val)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'f1': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_probs),
        'avg_precision': average_precision_score(y_val, y_probs)
    }
    
    # Log metrics
    experiment.log_metrics(metrics)
    
    # Classification report
    clf_report = classification_report(y_val, y_pred, output_dict=True)
    experiment.log_metrics({
        'precision': clf_report['weighted avg']['precision'],
        'recall': clf_report['weighted avg']['recall'],
        'f1_weighted': clf_report['weighted avg']['f1-score']
    })
    
    # Confusion matrix plot
    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    experiment.log_figure(figure_name="Confusion Matrix", figure=plt)
    plt.close()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_val.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
    plt.title('Top 20 Feature Importances')
    experiment.log_figure(figure_name="Feature Importance", figure=plt)
    plt.close()
    
    return metrics

def save_artifacts(model: RandomForestClassifier, encoder: OneHotEncoder,
                  file_path: str = 'artifacts') -> None:
    """Save model and encoder to disk"""
    import os
    os.makedirs(file_path, exist_ok=True)
    
    joblib.dump(model, f'{file_path}/fraud_detection_model.joblib')
    joblib.dump(encoder, f'{file_path}/feature_encoder.joblib')
    logger.info(f"Artifacts saved to {file_path}/")

def main():
    # Initialize Comet ML experiment
    experiment = comet_ml.start(project_name="Fraud_Detection")
    experiment.add_tag("RF_HalvingGridSearch_SMOTE")
    
    try:

        # 1. Load and preprocess data
        df = load_and_preprocess_data('data/transactions.csv')
        
        # 2. Feature encoding and train-test split
        X_train, X_val, y_train, y_val, encoder = encode_and_split_data(df)
        
        # 3. Apply SMOTE resampling
        X_train_resampled, y_train_resampled = resample_data(X_train, y_train)
        
        # 4. Model training
        best_model, best_params = train_model(X_train_resampled, y_train_resampled, experiment)
        
        # 5. Model evaluation
        metrics = evaluate_model(best_model, X_val, y_val, experiment)
        
        # 6. Save artifacts
        save_artifacts(best_model, encoder)
        
        logger.info(f"Training complete. Best params: {best_params}")
        logger.info(f"Validation metrics: {metrics}")
        
        experiment.log_other("status", "success")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        experiment.log_other("status", "failed")
        raise
    finally:
        experiment.end()

if __name__ == "__main__":
    main()