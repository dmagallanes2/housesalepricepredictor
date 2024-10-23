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
            lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=100000, value=10000)
            gr_liv_area = st.number_input("Living Area (sq ft)", min_value=500, max_value=10000, value=1500)
            year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000)
            
        with col2:
            overall_qual = st.slider("Overall Quality", min_value=1, max_value=10, value=7)
            neighborhood = st.selectbox("Neighborhood", sorted(st.session_state.model.data['Neighborhood'].unique()))
            house_style = st.selectbox("House Style", sorted(st.session_state.model.data['HouseStyle'].unique()))
        
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
                
                details = f"""
                • Living Area: {gr_liv_area:,} sq ft
                • Lot Size: {lot_area:,} sq ft
                • Quality Rating: {overall_qual}/10
                • Age: {age} years
                • Style: {house_style}
                • Location: {neighborhood}
                """
                st.markdown(details)
                
                # Market Analysis
                st.markdown("\n📈 **Market Analysis:**")
                similar_count = len(similar_props)
                price_min = similar_props['SalePrice'].min()
                price_avg = similar_props['SalePrice'].mean()
                price_max = similar_props['SalePrice'].max()
                
                analysis = f"""
                • Similar Properties: {similar_count}
                • Price Range in Area:
                - Minimum: ${price_min:,.2f}
                - Average: ${price_avg:,.2f}
                - Maximum: ${price_max:,.2f}
                """
                st.markdown(analysis)
                
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
                
                with col2:
                    # Similar Properties Pie Chart
                    total_properties = len(st.session_state.model.data)
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=[f'Similar\n({similar_count})', f'Other\n({total_properties - similar_count})'],
                            values=[similar_count, total_properties - similar_count],
                            marker_colors=['lightblue', 'gray']
                        )
                    ])
                    fig.update_layout(
                        title='Similar Properties Distribution',
                        title_x=0.5
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
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
                    
    except Exception as e:
        logger.error(f"Error in make_prediction: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def main():
    """Main application function"""
    # App title
    st.title("🏠 House Price Predictor")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Train Model", "Make Predictions"])
    
    with tab1:
        load_and_train()
        
    with tab2:
        make_prediction()
    
    # Add footer
    st.markdown("---")
    st.markdown("Built with Streamlit • [Source Code](https://github.com/dmagallanes2/housesalepricepredictor)")

if __name__ == "__main__":
    main()