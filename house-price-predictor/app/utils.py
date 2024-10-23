import pandas as pd
import numpy as np
from datetime import datetime
import logging

class DataProcessor:
    """Utility functions for data processing and validation"""
    
    @staticmethod
    def validate_data_structure(df):
        """
        Validate that the uploaded data has the required columns
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        
        Returns:
        tuple: (bool, str) - (is_valid, error_message)
        """
        required_columns = {
            'SalePrice',
            'LotArea',
            'GrLivArea',
            'OverallQual',
            'YearBuilt',
            'Neighborhood',
            'HouseStyle'
        }
        
        missing_cols = required_columns - set(df.columns)
        
        if missing_cols:
            return False, f"Missing required columns: {', '.join(missing_cols)}"
        return True, "Data structure valid"

    @staticmethod
    def preprocess_features(features_dict):
        """
        Preprocess input features for prediction
        
        Parameters:
        features_dict (dict): Raw input features
        
        Returns:
        dict: Processed features
        """
        processed = features_dict.copy()
        
        # Convert numeric fields
        numeric_fields = ['LotArea', 'GrLivArea', 'OverallQual', 'YearBuilt']
        for field in numeric_fields:
            if field in processed:
                processed[field] = float(processed[field])
                
        return processed

    @staticmethod
    def calculate_summary_stats(df):
        """
        Calculate summary statistics for the dataset
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        
        Returns:
        dict: Summary statistics
        """
        stats = {
            'total_properties': len(df),
            'avg_price': df['SalePrice'].mean(),
            'median_price': df['SalePrice'].median(),
            'price_std': df['SalePrice'].std(),
            'avg_area': df['GrLivArea'].mean(),
            'price_range': {
                'min': df['SalePrice'].min(),
                'max': df['SalePrice'].max()
            },
            'avg_age': datetime.now().year - df['YearBuilt'].mean(),
            'neighborhoods': df['Neighborhood'].nunique()
        }
        return stats

    @staticmethod
    def get_price_segments(df, n_segments=5):
        """
        Segment properties by price range
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        n_segments (int): Number of price segments
        
        Returns:
        pd.DataFrame: DataFrame with price segments
        """
        df = df.copy()
        df['PriceSegment'] = pd.qcut(df['SalePrice'], q=n_segments, labels=[
            'Entry Level',
            'Affordable',
            'Mid Range',
            'High End',
            'Luxury'
        ])
        return df

    @staticmethod
    def get_similar_properties(df, target_price, n_properties=5):
        """
        Find similar properties based on price
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        target_price (float): Target price point
        n_properties (int): Number of similar properties to return
        
        Returns:
        pd.DataFrame: Similar properties
        """
        df['price_diff'] = abs(df['SalePrice'] - target_price)
        similar = df.nsmallest(n_properties, 'price_diff')
        return similar.drop('price_diff', axis=1)

    @staticmethod
    def format_price(price):
        """
        Format price value for display
        
        Parameters:
        price (float): Price value
        
        Returns:
        str: Formatted price string
        """
        return f"${price:,.2f}"

    @staticmethod
    def calculate_price_per_sqft(df):
        """
        Calculate price per square foot
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        
        Returns:
        pd.Series: Price per square foot
        """
        return df['SalePrice'] / df['GrLivArea']