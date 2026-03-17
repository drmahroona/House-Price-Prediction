import streamlit as st
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from house_price_model import HousePricePredictor
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.markdown("""
    <style>
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }
    .price-tag {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .positive-factor {
        color: #28a745;
        font-weight: bold;
    }
    .negative-factor {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

if 'predictor' not in st.session_state:
    st.session_state.predictor = HousePricePredictor()
if 'model_ready' not in st.session_state:
    st.session_state.model_ready = False

@st.cache_resource
def initialize_model():
    """Initialize with correctly correlated training data"""
    
    if os.path.exists('house_price_model_weights.pth') and os.path.exists('house_price_model_components.pkl'):
        try:
            st.session_state.predictor.load_model()
            st.session_state.model_ready = True
            return "loaded"
        except Exception as e:
            st.warning(f"Could not load existing model: {e}")
    
    np.random.seed(42)
    n_samples = 2000
    
    cities = ['Seattle', 'Bellevue', 'Redmond', 'Kirkland', 'Renton', 'Issaquah']
    
    city_base = {
        'Bellevue': 950000,    # Most expensive
        'Kirkland': 850000,
        'Redmond': 800000,
        'Seattle': 750000,
        'Issaquah': 700000,
        'Renton': 550000       # Least expensive
    }
    
    data = []
    
    for i in range(n_samples):
        city = np.random.choice(cities)
        base = city_base[city]
        
        sqft = np.random.normal(2200, 500)
        sqft = max(1000, min(4500, sqft))
        
        bedrooms = int(np.clip(np.round(sqft / 500 + np.random.normal(0, 0.3)), 2, 6))
        bathrooms = float(np.clip(np.round(sqft / 350 + np.random.normal(0, 0.2)), 1.5, 4.5))
        
        room_factor = 1.0
        room_factor += (bedrooms - 3) * 0.05   
        room_factor += (bathrooms - 2.5) * 0.04 
        
        condition = np.random.randint(1, 6)
        condition_factor = 0.8 + (condition - 1) * 0.1  
        
        view = np.random.randint(0, 5)
        view_factor = 1.0 + view * 0.03  
        
        waterfront = 1 if np.random.random() < 0.03 else 0
        waterfront_factor = 1.3 if waterfront == 1 else 1.0  
        
        yr_built = np.random.randint(1960, 2023)
        age = 2024 - yr_built
        age_factor = 1.2 - (age / 200)  
        
        yr_renovated = 0
        renovation_factor = 1.0
        if np.random.random() < 0.3:
            yr_renovated = np.random.randint(max(yr_built, 2000), 2023)
            years_since = 2024 - yr_renovated
            renovation_factor = 1.15 - (years_since / 100)  

        size_factor = (sqft / 2000) ** 1.1  

        price = (base * 
                size_factor * 
                room_factor * 
                condition_factor * 
                view_factor * 
                waterfront_factor * 
                age_factor * 
                renovation_factor)
        
        price *= np.random.normal(1, 0.03)
        price = int(max(300000, min(2000000, price)))
        
        data.append({
            'price': price,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqft_living': int(sqft),
            'sqft_lot': int(sqft * np.random.uniform(2, 5)),
            'floors': float(np.random.choice([1, 1.5, 2, 2.5, 3])),
            'waterfront': waterfront,
            'view': view,
            'condition': condition,
            'sqft_above': int(sqft * 0.8),
            'sqft_basement': int(sqft * 0.4 if np.random.random() > 0.4 else 0),
            'yr_built': yr_built,
            'yr_renovated': yr_renovated,
            'city': city
        })
    
    df = pd.DataFrame(data)
    
    st.sidebar.markdown("### 📊 Data Verification")
    st.sidebar.write("**Correct Correlations:**")
    
    corr_bed = df['bedrooms'].corr(df['price'])
    corr_bath = df['bathrooms'].corr(df['price'])
    corr_condition = df['condition'].corr(df['price'])
    corr_view = df['view'].corr(df['price'])
    corr_waterfront = df['waterfront'].corr(df['price'])
    corr_year = df['yr_built'].corr(df['price'])
    
    st.sidebar.write(f"Bedrooms vs Price: {corr_bed:.2f} (should be positive)")
    st.sidebar.write(f"Bathrooms vs Price: {corr_bath:.2f} (should be positive)")
    st.sidebar.write(f"Condition vs Price: {corr_condition:.2f} (should be positive)")
    st.sidebar.write(f"View vs Price: {corr_view:.2f} (should be positive)")
    st.sidebar.write(f"Waterfront vs Price: {corr_waterfront:.2f} (should be positive)")
    st.sidebar.write(f"Year Built vs Price: {corr_year:.2f} (should be positive)")
    
    city_prices = df.groupby('city')['price'].mean().sort_values(ascending=False)
    st.sidebar.write("\n**City Price Order:**")
    for city, price in city_prices.items():
        st.sidebar.write(f"{city}: ${price:,.0f}")
    
    with st.spinner("Training model with correct correlations..."):
        st.session_state.predictor.train(df, epochs=100)
        st.session_state.predictor.save_model()
        st.session_state.model_ready = True
    
    return "trained"

if not st.session_state.model_ready:
    status = initialize_model()
    if status:
        st.rerun()

st.title("🏠 House Price Predictor")
st.markdown("Enter your property details for an instant estimate")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        city = st.selectbox(
            "📍 City",
            ['Bellevue', 'Kirkland', 'Redmond', 'Seattle', 'Issaquah', 'Renton'],
            index=3 
        )
        
        sqft = st.number_input(
            "📏 Living Area (sq ft)",
            min_value=500, max_value=5000, value=2200, step=50
        )
        
        bedrooms = st.slider(
            "🛏️ Bedrooms",
            min_value=1, max_value=6, value=3
        )
        
        bathrooms = st.slider(
            "🚿 Bathrooms",
            min_value=1.0, max_value=4.5, value=2.5, step=0.5
        )
    
    with col2:
        yr_built = st.number_input(
            "📅 Year Built",
            min_value=1900, max_value=2024, value=1995
        )
        
        yr_renovated = st.number_input(
            "🔨 Year Renovated",
            min_value=0, max_value=2024, value=0,
            help="Enter 0 if never renovated"
        )
        
        condition = st.slider(
            "⭐ Condition (1-5)",
            min_value=1, max_value=5, value=3,
            help="5 = Excellent, 1 = Poor"
        )
        
        view = st.slider(
            "👁️ View Rating (0-4)",
            min_value=0, max_value=4, value=0,
            help="4 = Excellent view"
        )
    
    waterfront = st.checkbox("💧 Waterfront Property")
    
    submitted = st.form_submit_button("🔮 Calculate Price", type="primary", use_container_width=True)

if submitted:
    input_data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft,
        'sqft_lot': sqft * 3,
        'floors': 2.0,
        'waterfront': 1 if waterfront else 0,
        'view': view,
        'condition': condition,
        'sqft_above': int(sqft * 0.8),
        'sqft_basement': 0,
        'yr_built': yr_built,
        'yr_renovated': yr_renovated,
        'city': city
    }
    
    result = st.session_state.predictor.predict(input_data)
    price = result['predicted_price']
    
    if price >= 1_000_000:
        price_str = f"${price/1_000_000:.2f}M"
    else:
        price_str = f"${price:,.0f}"
    
    st.markdown(f"""
    <div class="prediction-box">
        <h2>Estimated Value</h2>
        <div class="price-tag">{price_str}</div>
        <p>Range: ${result['lower_bound']:,.0f} - ${result['upper_bound']:,.0f}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show factor analysis
    st.subheader("📊 Price Factors")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Price per sq ft", f"${price/sqft:,.0f}")
        if bedrooms > 3:
            st.markdown(f"<span class='positive-factor'>✓ Extra bedrooms (+{((bedrooms-3)*5)}%)</span>", unsafe_allow_html=True)
        elif bedrooms < 3:
            st.markdown(f"<span class='negative-factor'>⚠ Fewer bedrooms (-{((3-bedrooms)*5)}%)</span>", unsafe_allow_html=True)
    
    with col2:
        if condition > 3:
            st.markdown(f"<span class='positive-factor'>✓ Excellent condition (+{((condition-3)*10)}%)</span>", unsafe_allow_html=True)
        elif condition < 3:
            st.markdown(f"<span class='negative-factor'>⚠ Below average condition (-{((3-condition)*10)}%)</span>", unsafe_allow_html=True)
        
        if view > 0:
            st.markdown(f"<span class='positive-factor'>✓ Good view (+{view*3}%)</span>", unsafe_allow_html=True)
    
    with col3:
        if waterfront:
            st.markdown(f"<span class='positive-factor'>✓ Waterfront (+30%)</span>", unsafe_allow_html=True)
        
        age = 2024 - yr_built
        if age < 10:
            st.markdown(f"<span class='positive-factor'>✓ New construction (+{(10-age)*2}%)</span>", unsafe_allow_html=True)
        elif age > 50:
            st.markdown(f"<span class='negative-factor'>⚠ Older home (-{min(30, age/2)}%)</span>", unsafe_allow_html=True)
    
    if yr_renovated > 0:
        years_since = 2024 - yr_renovated
        if years_since < 5:
            st.success(f"✨ Recently renovated - adds significant value!")
        elif years_since < 10:
            st.info(f"🏠 Renovated {years_since} years ago - still adds value")

st.markdown("---")
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: transparent;
    color: gray;
    text-align: center;
    padding: 10px;
    font-size: 14px;
}
</style>

<div class="footer">
    Made by <b>Dr. Mahroona Laraib</b> |
    <a href="https://github.com/drmahroona" target="_blank">GitHub</a>
</div>
""", unsafe_allow_html=True)