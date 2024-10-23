import streamlit as st
import pandas as pd
from app.model import HousePriceModel
from app.utils import DataProcessor
from app.visualizations import DataVisualizer
import logging
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.model_selection import cross_val_score, KFold, learning_curve, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

# Configure page
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = HousePriceModel()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_train():
    """Load data and train model"""
    st.markdown("""
    ### Step 1: Upload Training Data
    Upload the house sales dataset (train.csv) to begin. This will be used to train the model.
    """)
    try:
        # Load and validate data
        uploaded_file = st.file_uploader("Upload your housing dataset (CSV)", type="csv")
        
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            
            # Validate data structure
            is_valid, message = DataProcessor.validate_data_structure(data)
            
            if not is_valid:
                st.error(message)
                return
            
            # Load data into model
            st.session_state.model.load_data(data)
            st.session_state.model.clean_data()
            st.session_state.data_loaded = True
            
            # Display data summary
            st.success("Data loaded successfully!")
            stats = DataProcessor.calculate_summary_stats(data)
            
            # Display key metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Properties", f"{stats['total_properties']:,}")
            col2.metric("Average Price", f"${stats['avg_price']:,.2f}")
            col3.metric("Average Area", f"{stats['avg_area']:,.0f} sq ft")
            col4.metric("Neighborhoods", stats['neighborhoods'])
            
            # Train model button
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    rmse, r2 = st.session_state.model.train_model()
                    st.session_state.model_trained = True
                    st.success(f"Model trained successfully! RMSE: ${rmse:,.2f}, R²: {r2:.4f}")
                    
                    # Show feature importance
                    importance_df = st.session_state.model.get_feature_importance()
                    st.plotly_chart(DataVisualizer.plot_feature_importance(importance_df))
            
            # Show initial visualizations
            if st.session_state.data_loaded:
                st.header("📊 Data Exploration and Visualization")
                st.markdown("Explore the dataset through various visualizations to understand price distributions, relationships, and trends.")
                with st.expander("View Data Analysis", expanded=True):
                    # Create tabs for different visualizations
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "Price Distribution", 
                        "Price vs Living Area", 
                        "Neighborhood Analysis",
                        "Price Percentiles",
                        "Area vs Price Trend"
                    ])
                    
                    with tab1:
                        # House Price Distribution
                        st.markdown("This histogram shows the distribution of house prices in the dataset.")
                        fig = px.histogram(
                            data,
                            x='SalePrice',
                            nbins=50,
                            title='House Price Distribution'
                        )
                        fig.update_layout(
                            xaxis_title="Price ($)",
                            yaxis_title="Count",
                            title_x=0.5
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        # Price vs Living Area Scatter
                        st.markdown("This scatter plot shows the relationship between living area and house prices.")
                        fig = px.scatter(
                            data,
                            x='GrLivArea',
                            y='SalePrice',
                            title='Price vs Living Area'
                        )
                        fig.update_layout(
                            xaxis_title="Living Area (sq ft)",
                            yaxis_title="Price ($)",
                            title_x=0.5
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        # Price Distribution by Neighborhood
                        st.markdown("This box plot shows how house prices vary across different neighborhoods.")
                        fig = px.box(
                            data,
                            x='Neighborhood',
                            y='SalePrice',
                            title='Price Distribution by Neighborhood'
                        )
                        fig.update_layout(
                            xaxis_tickangle=-45,
                            xaxis_title="Neighborhood",
                            yaxis_title="Price ($)",
                            title_x=0.5
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab4:
                        # Price Distribution with Percentiles
                        st.markdown("This plot shows the price distribution with key percentile markers.")
                        prices_sorted = sorted(data['SalePrice'])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(range(len(prices_sorted))),
                            y=prices_sorted,
                            mode='lines',
                            name='Prices'
                        ))
                        
                        # Add percentile lines
                        percentiles = [25, 50, 75]
                        colors = ['red', 'green', 'blue']
                        names = ['25th percentile', 'Median', '75th percentile']
                        
                        for p, c, n in zip(percentiles, colors, names):
                            value = np.percentile(prices_sorted, p)
                            fig.add_hline(
                                y=value,
                                line_dash="dash",
                                line_color=c,
                                annotation_text=n,
                                annotation_position="right"
                            )
                        
                        fig.update_layout(
                            title='Price Distribution with Percentiles',
                            xaxis_title="Property Index",
                            yaxis_title="Price ($)",
                            title_x=0.5
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab5:
                        # Area vs Price with Trend
                        st.markdown("This scatter plot shows the relationship between area and price with a trend line.")
                        fig = px.scatter(
                            data,
                            x='GrLivArea',
                            y='SalePrice',
                            title='Area vs Price with Trend'
                        )
                        
                        # Add trend line
                        z = np.polyfit(data['GrLivArea'], data['SalePrice'], 1)
                        p = np.poly1d(z)
                        fig.add_trace(go.Scatter(
                            x=sorted(data['GrLivArea']),
                            y=p(sorted(data['GrLivArea'])),
                            mode='lines',
                            name='Trend',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig.update_layout(
                            xaxis_title="Living Area (sq ft)",
                            yaxis_title="Price ($)",
                            title_x=0.5
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
    except Exception as e:
        logger.error(f"Error in load_and_train: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def make_prediction():
    """Create prediction interface and make predictions"""
    try:
        if not st.session_state.model_trained:
            st.warning("Please load data and train the model first!")
            return
            
        st.subheader("Enter Property Details")
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)

        with col1:
            lot_area = st.slider(
                "Lot Area (sq ft)", 
                min_value=1000, 
                max_value=100000, 
                value=10000,
                step=100,
                help="Total square footage of the property lot"
            )
            gr_liv_area = st.slider(
                "Living Area (sq ft)", 
                min_value=500, 
                max_value=10000, 
                value=1500,
                step=100,
                help="Above ground living area in square feet"
            )
            year_built = st.slider(
                "Year Built", 
                min_value=1800, 
                max_value=2024, 
                value=2000,
                step=1,
                help="Original construction year of the house"
            )

        with col2:
            overall_qual = st.slider(
                "Overall Quality", 
                min_value=1, 
                max_value=10, 
                value=7,
                help="Overall material and finish quality (1=Poor, 10=Excellent)"
            )
            neighborhood = st.selectbox(
                "Neighborhood", 
                sorted(st.session_state.model.data['Neighborhood'].unique())
            )
            house_style = st.selectbox(
                "House Style", 
                sorted(st.session_state.model.data['HouseStyle'].unique())
            )
        
        # Make prediction
        if st.button("Predict Price"):
            features = {
                'LotArea': lot_area,
                'GrLivArea': gr_liv_area,
                'OverallQual': overall_qual,
                'YearBuilt': year_built,
                'Neighborhood': neighborhood,
                'HouseStyle': house_style
            }
            
            with st.spinner("Making prediction..."):
                prediction = st.session_state.model.predict(features)
                similar_props = DataProcessor.get_similar_properties(
                    st.session_state.model.data,
                    prediction
                )
                
                # Display formatted prediction results
                st.markdown("═══ House Price Prediction ═══")
                st.markdown(f"💰 **Predicted Price:** ${prediction:,.2f}")

                st.markdown("\n📊 **Property Details:**")
                current_year = datetime.now().year
                age = current_year - year_built

                # Format each detail on its own line
                st.markdown("""
                - Living Area: {} sq ft

                - Lot Size: {} sq ft

                - Quality Rating: {}/10

                - Age: {} years

                - Style: {}

                - Location: {}
                """.format(
                    f"{gr_liv_area:,}",
                    f"{lot_area:,}",
                    overall_qual,
                    age,
                    house_style,
                    neighborhood
                ))

                # Calculate market analysis values
                similar_count = len(similar_props)
                price_min = similar_props['SalePrice'].min()
                price_avg = similar_props['SalePrice'].mean()
                price_max = similar_props['SalePrice'].max()

                # Market Analysis with better spacing
                st.markdown("\n📈 **Market Analysis:**")
                st.markdown(f"• Similar Properties: {similar_count}")
                st.markdown("\n• Price Range in Area:")
                st.markdown(f"""
                - Minimum: ${price_min:,.2f}

                - Average: ${price_avg:,.2f}

                - Maximum: ${price_max:,.2f}
                """)
                
                # Create three visualization columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    # Price Comparison Bar Chart
                    prices = [price_min, prediction, price_avg, price_max]
                    labels = ['Minimum', 'Predicted', 'Average', 'Maximum']
                    fig = go.Figure(data=[
                        go.Bar(x=labels, y=prices, marker_color=['lightblue', 'green', 'blue', 'darkblue'])
                    ])
                    fig.update_layout(
                        title='Price Comparison',
                        yaxis_title='Price ($)',
                        title_x=0.5
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("*This graph compares your predicted price with the minimum, average, and maximum prices of similar properties.*")

                with col2:
                    # Quality vs Price Comparison
                    quality_prices = st.session_state.model.data.groupby('OverallQual')['SalePrice'].mean().reset_index()
                    fig = go.Figure(data=[
                        go.Scatter(
                            x=quality_prices['OverallQual'],
                            y=quality_prices['SalePrice'],
                            mode='lines+markers',
                            name='Average Price'
                        )
                    ])
                    fig.add_trace(go.Scatter(
                        x=[overall_qual],
                        y=[prediction],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='star'),
                        name='Your Property'
                    ))
                    fig.update_layout(
                        title='Price vs Quality Rating',
                        xaxis_title='Overall Quality',
                        yaxis_title='Price ($)',
                        title_x=0.5
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("*This graph shows how your predicted price compares to average prices for different quality ratings.*")

                with col3:
                    # Neighborhood Price Distribution
                    neighborhood_data = st.session_state.model.data[
                        st.session_state.model.data['Neighborhood'] == neighborhood
                    ]
                    fig = px.histogram(
                        neighborhood_data,
                        x='SalePrice',
                        title=f'Price Distribution in {neighborhood}'
                    )
                    fig.add_vline(
                        x=prediction,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Predicted"
                    )
                    fig.update_layout(
                        title_x=0.5,
                        xaxis_title="Price ($)",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown("*This histogram shows where your predicted price falls within the selected neighborhood's price distribution.*")
                    
    except Exception as e:
        logger.error(f"Error in make_prediction: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        
def show_validation_results():
    """Display model validation results and visualizations"""

    try:
        if not st.session_state.model_trained:
            st.warning("Please train the model first!")
            return

        st.header("🔍 Model Validation Suite")
        st.markdown("""
        This section shows detailed analysis of model performance including cross-validation scores,
        error metrics, and various visualizations of model behavior.
        """)
        
        # Get model and data from session state
        model = st.session_state.model.model
        data = st.session_state.model.data
        
        # Prepare features
        X = pd.get_dummies(data.drop('SalePrice', axis=1))
        y = data['SalePrice']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 1. Cross-Validation
        st.subheader("1. Cross-Validation Results")
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
        
        st.write(f"Cross-validation R² scores: {cv_scores}")
        st.write(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # 2. Performance Metrics
        st.subheader("2. Performance Metrics")

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        col1, col2 = st.columns(2)
        with col1:
            st.write("Training Set Metrics:")
            st.write(f"RMSE: ${np.sqrt(mean_squared_error(y_train, y_pred_train)):,.2f}")
            st.write(f"R²: {r2_score(y_train, y_pred_train):.4f}")
            st.write(f"MAE: ${mean_absolute_error(y_train, y_pred_train):,.2f}")
        
        with col2:
            st.write("Test Set Metrics:")
            st.write(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred_test)):,.2f}")
            st.write(f"R²: {r2_score(y_test, y_pred_test):.4f}")
            st.write(f"MAE: ${mean_absolute_error(y_test, y_pred_test):,.2f}")

        # 3. Residual Analysis
        st.subheader("3. Residual Analysis")
        
        residuals = y_test - y_pred_test
        
        st.write(f"Mean of residuals: ${residuals.mean():,.2f}")
        st.write(f"Standard deviation of residuals: ${residuals.std():,.2f}")
        st.write(f"Skewness of residuals: {stats.skew(residuals):.4f}")
        st.write(f"Kurtosis of residuals: {stats.kurtosis(residuals):.4f}")

        # Create three columns for visualizations
        st.subheader("Model Performance Visualizations")
        col1, col2, col3 = st.columns(3)

        with col1:
            # Cross-validation scores distribution
            fig = go.Figure()
            fig.add_trace(go.Box(y=cv_scores, name='R² Scores'))
            fig.update_layout(
                title='Cross-validation R² Scores Distribution',
                yaxis_title='R² Score',
                title_x=0.5
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Feature importance
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)
            
            fig = go.Figure(go.Bar(
                x=importance_df['importance'],
                y=importance_df['feature'],
                orientation='h'
            ))
            fig.update_layout(
                title='Top 10 Feature Importance',
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                title_x=0.5
            )
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            # Learning curve
            train_sizes = np.linspace(0.1, 1.0, 10)
            train_sizes, train_scores, test_scores = learning_curve(
                model, X_train, y_train,
                train_sizes=train_sizes,
                cv=5,
                scoring='r2'
            )

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=train_scores.mean(axis=1),
                name='Training score'
            ))
            fig.add_trace(go.Scatter(
                x=train_sizes,
                y=test_scores.mean(axis=1),
                name='Cross-validation score'
            ))
            fig.update_layout(
                title='Learning Curve',
                xaxis_title='Training Examples',
                yaxis_title='R² Score',
                title_x=0.5
            )
            st.plotly_chart(fig, use_container_width=True)

        # 5. Prediction Error Analysis
        st.subheader("5. Prediction Error Analysis")
        error_percentage = (np.abs(y_test - y_pred_test) / y_test) * 100
        
        st.write(f"Mean Absolute Percentage Error: {error_percentage.mean():.2f}%")
        st.write(f"Median Percentage Error: {np.median(error_percentage):.2f}%")
        st.write(f"Predictions within 10% of actual value: {(error_percentage <= 10).mean() * 100:.2f}%")
        st.write(f"Predictions within 20% of actual value: {(error_percentage <= 20).mean() * 100:.2f}%")

        # Acceptance Criteria
        st.subheader("Model Acceptance Criteria")
        criteria_met = {
            'R² Score > 0.7': r2_score(y_test, y_pred_test) >= 0.7,
            'RMSE < $50,000': np.sqrt(mean_squared_error(y_test, y_pred_test)) <= 50000,
            '80% predictions within 20%': (error_percentage <= 20).mean() * 100 >= 80
        }
        
        for criterion, passed in criteria_met.items():
            st.write(f"{criterion}: {'✅ PASSED' if passed else '❌ FAILED'}")
        
        if all(criteria_met.values()):
            st.success("Model ACCEPTED for deployment")
        else:
            st.error("Model NEEDS IMPROVEMENT before deployment")
            
    except Exception as e:
        logger.error(f"Error in validation suite: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def main():
    """Main application function"""
    # App title
    st.title("🏠 House Price Predictor")
    st.markdown("""
    This app predicts house prices based on various features. Follow these steps:
    1. Upload the training data (train.csv)
    2. Train the model
    3. Make predictions on new properties
    """)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Train Model", "Make Predictions", "Model Validation"])
    
    with tab1:
        load_and_train()
        
    with tab2:
        make_prediction()
        
    with tab3:
        show_validation_results()
    
    # Add footer
    st.markdown("---")
    st.markdown("Built with Streamlit • [Source Code](https://github.com/dmagallanes2/housesalepricepredictor)")

if __name__ == "__main__":
    main()