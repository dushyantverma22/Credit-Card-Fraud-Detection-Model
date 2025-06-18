import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, confusion_matrix,
                           classification_report, roc_auc_score)
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Dict
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
    def __init__(self, model_path: str = "artifacts"):
        """
        Initialize prediction pipeline with paths to saved artifacts
        
        Args:
            model_path: Directory containing saved model and encoder
        """
        self.model_path = model_path
        self.model = None
        self.encoder = None
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """Load trained model and feature encoder"""
        try:
            self.model = joblib.load(os.path.join(self.model_path, "fraud_detection_model.joblib"))
            self.encoder = joblib.load(os.path.join(self.model_path, "feature_encoder.joblib"))
            logger.info("✅ Model and encoder loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load artifacts: {str(e)}")
            raise

    def preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess raw transaction data (same as training pipeline)
        
        Args:
            raw_data: Raw transaction DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # [Include your complete data_preprocessing() function here]
        # Example placeholder implementation:
        processed_data = raw_data.copy()
        processed_data.set_index("trans_num", inplace=True)
        # ... rest of preprocessing steps ...
        return processed_data

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using the saved encoder
        
        Args:
            df: Preprocessed DataFrame
            
        Returns:
            DataFrame with encoded features
        """
        # Define columns (must match training)
        cat_cols = ['time_bucket', 'category', 'amount_bkt', 'population_bkt', 'age_bkt']
        drop_cols = cat_cols + ["trans_date", "cc_num", "gender", "state", "age"]
        
        # Transform using saved encoder
        X_encoded = self.encoder.transform(df[cat_cols])
        encoded_feat_names = self.encoder.get_feature_names_out(cat_cols)
        
        # Create encoded DataFrame
        encoded_df = pd.DataFrame(X_encoded, columns=encoded_feat_names, index=df.index)
        data_processed = df.drop(columns=drop_cols)
        final_features = pd.concat([data_processed, encoded_df], axis=1)
        
        # Ensure column order matches training
        expected_columns = [
            'latitudinal_distance', 'longitudinal_distance', 'gender_encod',
            'time_bucket_6AM-12PM', 'time_bucket_12PM-6PM', 'time_bucket_6PM-12AM',
            # ... include all your expected feature columns ...
        ]
        
        return final_features.reindex(columns=expected_columns, fill_value=0)

    def predict(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate fraud predictions for new transactions
        
        Args:
            new_data: Raw transaction DataFrame
            
        Returns:
            DataFrame with original data + predictions
        """
        try:
            # 1. Preprocess
            processed_data = self.preprocess_data(new_data)
            
            # 2. Feature encoding
            X_new = self.encode_features(processed_data)
            
            # 3. Generate predictions
            predictions = self.model.predict(X_new)
            probabilities = self.model.predict_proba(X_new)[:, 1]
            
            # 4. Add to output
            processed_data['predicted_fraud'] = predictions
            processed_data['fraud_probability'] = probabilities
            
            logger.info(f"Generated predictions for {len(new_data)} transactions")
            return processed_data
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise

    def evaluate_predictions(self, df: pd.DataFrame, y_true: pd.Series) -> Dict:
        """
        Evaluate model performance on labeled data
        
        Args:
            df: DataFrame containing 'predicted_fraud' column
            y_true: True labels (0/1)
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = df['predicted_fraud']
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, df['fraud_probability'])
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Non-Fraud', 'Predicted Fraud'],
                    yticklabels=['Actual Non-Fraud', 'Actual Fraud'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Non-Fraud', 'Fraud']))
        
        return metrics

def main():
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(model_path="artifacts")
    
    # Load new data (example)
    new_transactions = pd.read_csv("data/new_transactions.csv")
    
    # Generate predictions
    results = pipeline.predict(new_transactions)
    
    # Save results
    results.to_csv("predictions_with_fraud_flags.csv")
    logger.info("Predictions saved to predictions_with_fraud_flags.csv")
    
    # If you have true labels for evaluation
    if 'is_fraud' in new_transactions.columns:
        y_true = new_transactions['is_fraud']
        metrics = pipeline.evaluate_predictions(results, y_true)
        print("\nEvaluation Metrics:")
        print(pd.DataFrame([metrics]).T)

if __name__ == "__main__":
    main()