import streamlit as st
import pandas as pd
from app.model import HousePriceModel
from app.utils import DataProcessor
from app.visualizations import DataVisualizer
import logging

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
                    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Neighborhood Analysis", "Year Analysis"])
                    
                    with tab1:
                        st.plotly_chart(DataVisualizer.plot_price_distribution(data))
                    
                    with tab2:
                        st.plotly_chart(DataVisualizer.plot_neighborhood_prices(data))
                    
                    with tab3:
                        st.plotly_chart(DataVisualizer.plot_year_built_analysis(data))
                        
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
                
                # Show prediction
                st.success(f"Predicted Price: ${prediction:,.2f}")
                
                # Show visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(DataVisualizer.plot_price_distribution(
                        st.session_state.model.data,
                        prediction
                    ))
                    
                with col2:
                    st.plotly_chart(DataVisualizer.plot_price_vs_area(
                        st.session_state.model.data,
                        prediction,
                        gr_liv_area
                    ))
                
                # Show similar properties
                similar_props = DataProcessor.get_similar_properties(
                    st.session_state.model.data,
                    prediction
                )
                
                with st.expander("View Similar Properties"):
                    st.dataframe(similar_props)
                    
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
    st.markdown("Built with Streamlit • [Source Code](https://github.com/yourusername/house-price-predictor)")

if __name__ == "__main__":
    main()