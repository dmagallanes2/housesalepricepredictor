import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class DataVisualizer:
    """Create interactive visualizations for Streamlit app"""
    
    @staticmethod
    def plot_price_distribution(df, prediction=None):
        """
        Create price distribution plot
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        prediction (float, optional): Predicted price to overlay
        
        Returns:
        plotly.graph_objects.Figure: Interactive plot
        """
        fig = px.histogram(
            df,
            x='SalePrice',
            nbins=50,
            title='House Price Distribution',
            labels={'SalePrice': 'Price ($)', 'count': 'Number of Properties'}
        )
        
        # Add prediction line if provided
        if prediction:
            fig.add_vline(
                x=prediction,
                line_dash="dash",
                line_color="red",
                annotation_text="Predicted Price",
                annotation_position="top"
            )
            
        fig.update_layout(
            showlegend=True,
            xaxis_title="Price ($)",
            yaxis_title="Count",
            title_x=0.5
        )
        
        return fig

    @staticmethod
    def plot_feature_importance(importance_df):
        """
        Create feature importance plot
        
        Parameters:
        importance_df (pd.DataFrame): Feature importance DataFrame
        
        Returns:
        plotly.graph_objects.Figure: Interactive plot
        """
        fig = px.bar(
            importance_df,
            x='importance',
            y='feature',
            orientation='h',
            title='Top Feature Importance'
        )
        
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            title_x=0.5,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig

    @staticmethod
    def plot_price_vs_area(df, prediction=None, pred_area=None):
        """
        Create price vs. area scatter plot
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        prediction (float, optional): Predicted price
        pred_area (float, optional): Area for prediction point
        
        Returns:
        plotly.graph_objects.Figure: Interactive plot
        """
        fig = px.scatter(
            df,
            x='GrLivArea',
            y='SalePrice',
            title='Price vs. Living Area',
            labels={
                'GrLivArea': 'Living Area (sq ft)',
                'SalePrice': 'Price ($)'
            }
        )
        
        # Add prediction point if provided
        if prediction and pred_area:
            fig.add_scatter(
                x=[pred_area],
                y=[prediction],
                mode='markers',
                marker=dict(size=12, color='red', symbol='star'),
                name='Prediction',
                hovertemplate='Predicted Price: $%{y:,.2f}<br>Area: %{x} sq ft'
            )
            
        fig.update_layout(title_x=0.5)
        return fig

    @staticmethod
    def plot_neighborhood_prices(df):
        """
        Create neighborhood price comparison plot
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        
        Returns:
        plotly.graph_objects.Figure: Interactive plot
        """
        neighborhood_stats = df.groupby('Neighborhood')['SalePrice'].agg(['mean', 'count']).reset_index()
        neighborhood_stats = neighborhood_stats.sort_values('mean', ascending=True)
        
        fig = px.bar(
            neighborhood_stats,
            x='Neighborhood',
            y='mean',
            title='Average Price by Neighborhood',
            labels={'mean': 'Average Price ($)', 'Neighborhood': 'Neighborhood'},
            text='count'
        )
        
        fig.update_traces(
            texttemplate='%{text} properties',
            textposition='outside'
        )
        
        fig.update_layout(
            xaxis_tickangle=-45,
            title_x=0.5,
            height=500
        )
        
        return fig

    @staticmethod
    def create_metrics_dashboard(df, prediction=None):
        """
        Create dashboard with key metrics
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        prediction (float, optional): Predicted price
        """
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Price",
                f"${df['SalePrice'].mean():,.2f}",
                delta=f"${df['SalePrice'].std():,.2f} σ"
            )
            
        with col2:
            st.metric(
                "Average Area",
                f"{df['GrLivArea'].mean():,.0f} sq ft",
                delta=f"{df['GrLivArea'].std():,.0f} sq ft σ"
            )
            
        with col3:
            if prediction:
                percentile = (df['SalePrice'] < prediction).mean() * 100
                st.metric(
                    "Prediction Percentile",
                    f"{percentile:.1f}%",
                    delta="vs market"
                )

    @staticmethod
    def plot_year_built_analysis(df):
        """
        Create year built analysis plot
        
        Parameters:
        df (pd.DataFrame): Input DataFrame
        
        Returns:
        plotly.graph_objects.Figure: Interactive plot
        """
        year_stats = df.groupby('YearBuilt')['SalePrice'].mean().reset_index()
        
        fig = px.scatter(
            year_stats,
            x='YearBuilt',
            y='SalePrice',
            trendline="lowess",
            title='Average Price by Year Built',
            labels={
                'YearBuilt': 'Year Built',
                'SalePrice': 'Average Price ($)'
            }
        )
        
        fig.update_layout(title_x=0.5)
        return fig