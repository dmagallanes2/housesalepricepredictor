import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime

class HousePriceModel:
    """
    House Price Prediction Model
    Handles data processing, model training, and predictions
    """
    def __init__(self):
        """Initialize the model and components"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, data):
        """
        Load and prepare data for training
        
        Parameters:
        data (pd.DataFrame or str): DataFrame or path to CSV file
        
        Returns:
        pd.DataFrame: Processed DataFrame
        """
        try:
            if isinstance(data, str):
                self.data = pd.read_csv(data)
            else:
                self.data = data.copy()
                
            self.logger.info(f"Data loaded successfully: {self.data.shape}")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
            
    def clean_data(self):
        """
        Clean and prepare data for training
        
        Returns:
        pd.DataFrame: Cleaned DataFrame
        """
        try:
            # Handle missing numerical values
            for col in self.data.select_dtypes(include=[np.number]).columns:
                self.data[col] = self.data[col].fillna(self.data[col].mean())
            
            # Handle missing categorical values
            for col in self.data.select_dtypes(include=['object']).columns:
                self.data[col] = self.data[col].fillna('None')
            
            self.logger.info("Data cleaning completed")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error in data cleaning: {str(e)}")
            raise
            
    def prepare_features(self):
        """
        Prepare features for training
        
        Returns:
        tuple: X (features) and y (target) for model training
        """
        try:
            # Create feature matrix
            X = pd.get_dummies(self.data.drop('SalePrice', axis=1))
            y = self.data['SalePrice']
            
            # Store feature names for prediction
            self.feature_names = X.columns
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error in feature preparation: {str(e)}")
            raise
            
    def train_model(self):
        """
        Train the Random Forest model
        
        Returns:
        tuple: Training metrics (RMSE, R²)
        """
        try:
            # Prepare data
            X, y = self.prepare_features()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1  # Use all available cores
            )
            self.model.fit(X_train, y_train)
            
            # Calculate metrics
            predictions = self.model.predict(X_test)
            rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
            r2 = self.model.score(X_test, y_test)
            
            self.logger.info(f"Model trained successfully. RMSE: {rmse:.2f}, R²: {r2:.4f}")
            return rmse, r2
            
        except Exception as e:
            self.logger.error(f"Error in model training: {str(e)}")
            raise
            
    def predict(self, features):
        """
        Make price predictions for new data
        
        Parameters:
        features (dict): Property features for prediction
        
        Returns:
        float: Predicted price
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
                
            # Convert features to DataFrame
            input_df = pd.DataFrame([features])
            
            # Create dummy variables
            input_encoded = pd.get_dummies(input_df)
            
            # Ensure all model features are present
            for col in self.feature_names:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
                    
            # Ensure columns are in the correct order
            input_encoded = input_encoded[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(input_encoded)[0]
            
            self.logger.info(f"Prediction made successfully: ${prediction:,.2f}")
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def get_feature_importance(self, top_n=10):
        """
        Get the most important features
        
        Parameters:
        top_n (int): Number of top features to return
        
        Returns:
        pd.DataFrame: Top features and their importance scores
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
                
            # Get feature importance
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            })
            
            # Sort and return top features
            return importance.sort_values('importance', ascending=False).head(top_n)
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            raise