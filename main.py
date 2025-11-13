import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configure page
st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["About Us", "Introduction", "EDA", "Prediction"]
)

# About Us Page
if page == "About Us":
    st.title("� About Our Team")
    st.markdown("---")
    
    # Team Section
    st.markdown("## 🚀 **Our Team**")
    
    # Team Members
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 👨‍💻 **Ayuba Ngamarju Wabba**
        **Role: UI/UX Designer & Lead Developer**
        
        🎨 **Responsibilities:**
        - User Interface & User Experience Design
        - Frontend Development & Programming
        - Streamlit Dashboard Implementation
        - Interactive Visualization Development
        - Machine Learning Model Integration
        """)
        
        st.info("💡 **Expertise:** Python Programming, Streamlit, Data Visualization, UI/UX Design")
    
    with col2:
        st.markdown("""
        ### 📊 **Eko Kurniawan Foo Bin Arifin Foo**
        **Role: Data Analyst & Documentation Specialist**
        
        📈 **Responsibilities:**
        - Data Analysis & Interpretation
        - Research Documentation
        - Findings Analysis & Reporting
        - Statistical Analysis & Insights
        - Project Documentation Management
        """)
        
        st.info("💡 **Expertise:** Data Analysis, Statistical Research, Documentation, Market Research")
    
    st.markdown("---")
    
    # Mission Section
    st.markdown("## 🎯 **Our Mission**")
    
    st.success("""
    **Our mission is to make data science more accessible and insightful through interactive visualization 
    and machine learning integration. We aim to empower users to transform raw data into actionable 
    knowledge — supporting smarter, faster, and more informed decisions using intuitive visual tools.**
    """)
    
    st.markdown("---")
    
    # Team Collaboration
    st.markdown("## 🤝 **Team Collaboration**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎨 Design & Development**
        - Interactive dashboard design
        - User-friendly interface
        - Responsive visualizations
        - Machine learning integration
        """)
    
    with col2:
        st.markdown("""
        **📊 Analysis & Research**
        - Data exploration & cleaning
        - Statistical analysis
        - Pattern identification
        - Insight generation
        """)
    
    with col3:
        st.markdown("""
        **📝 Documentation & Reporting**
        - Project documentation
        - Findings analysis
        - Research methodology
        - Results interpretation
        """)
    
    st.markdown("---")
    
    # Project Information
    st.markdown("## � **Project Information**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📅 **Project Year**", "2025")
        st.caption("Academic Project")
    
    with col2:
        st.metric("🏗️ **Technology**", "Python")
        st.caption("Streamlit Framework")
    
    with col3:
        st.metric("📊 **Dataset Size**", "86,000+")
        st.caption("Airbnb Records")
    
    st.markdown("---")
    
    # Dataset Information
    st.markdown("## 📊 **Dataset Overview**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📈 Total Records", "86,000+")
        st.caption("Real Airbnb listings")
    
    with col2:
        st.metric("🌍 Geographic Scope", "Multiple Cities")
        st.caption("Including Toronto & more")
    
    with col3:
        st.metric("🏠 Property Types", "Various")
        st.caption("Entire homes, rooms, etc.")
    
    st.markdown("---")
    
    # Technology Stack
    st.markdown("## 🛠️ **Technology Stack**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Frontend & Visualization:**
        - 🎨 Streamlit (Interactive Dashboard)
        - 📊 Plotly (Dynamic Charts)
        - 🔥 Seaborn (Statistical Plots)
        - 📈 Matplotlib (Data Visualization)
        """)
    
    with col2:
        st.markdown("""
        **Data Science & ML:**
        - 🐍 Python (Core Language)
        - 🐼 Pandas (Data Manipulation)
        - 🤖 Scikit-learn (Machine Learning)
        - 🔢 NumPy (Numerical Computing)
        """)
    
    st.markdown("---")
    
    # Contact Section
    st.markdown("## 📞 **Project Information**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🎓 **Academic Project**\nStreamlit Data Dashboard")
    with col2:
        st.info("� **Year**\n2025")
    with col3:
        st.info("�️ **Built With**\nPython & Streamlit")

# Introduction Page
elif page == "Introduction":
    st.title("🚀 Project Introduction")
    st.markdown("---")
    
    # Project Overview
    st.markdown("""
    ## 🏠 Airbnb Data Visualization & Analysis Dashboard
    
    Data visualization is one of the cornerstones of modern data science, transforming the vast amounts of data 
    generated by today's systems into meaningful and actionable insights. In the era of Big Data, visualization 
    has evolved to not only display information but to enable interactive exploration and machine learning 
    (ML)-driven analysis.
    
    Our project leverages **Streamlit**, an open-source Python framework, to create an interactive dashboard 
    that combines data visualization with machine learning models. Using real **Airbnb listing data** with 
    over **86,000+ records**, we bridge the gap between data analysis and user experience — allowing users to 
    explore datasets, perform exploratory data analysis (EDA), and visualize predictive pricing results in real time.
    """)
    
    st.markdown("---")
    
    # Objectives Section
    st.markdown("## 🎯 **Project Objectives**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ✅ **Interactive Dashboard Design**  
        Design and develop an interactive Streamlit dashboard that visualizes Airbnb data dynamically.
        
        ✅ **Exploratory Data Analysis (EDA)**  
        Perform comprehensive EDA on Airbnb listings to uncover pricing patterns, geographic trends, and market insights.
        
        ✅ **Machine Learning Integration**  
        Apply supervised learning models for Airbnb price prediction that enhance data-driven decision-making.
        """)
    
    with col2:
        st.markdown("""
        ✅ **ML-Driven Visualization**  
        Integrate ML-driven visualization techniques that respond instantly to user interactions for price predictions.
        
        ✅ **Complex Data Simplification**  
        Demonstrate how visual analytics can simplify the understanding of large, complex Airbnb market data.
        
        ✅ **Real-World Application**  
        Provide actionable insights for travelers, hosts, and market researchers in the sharing economy.
        """)
    
    st.markdown("---")
    
    # Dataset Information
    st.markdown("## � **Dataset Overview**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📈 Total Records", "86,000+")
        st.caption("Real Airbnb listings")
    
    with col2:
        st.metric("🌍 Geographic Scope", "Multiple Cities")
        st.caption("Including Toronto & more")
    
    with col3:
        st.metric("🏠 Property Types", "Various")
        st.caption("Entire homes, rooms, etc.")
    
    st.markdown("---")
    
    # Technology Stack
    st.markdown("## 🛠️ **Technology Stack**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Frontend & Visualization:**
        - 🎨 Streamlit (Interactive Dashboard)
        - 📊 Plotly (Dynamic Charts)
        - 🔥 Seaborn (Statistical Plots)
        - 📈 Matplotlib (Data Visualization)
        """)
    
    with col2:
        st.markdown("""
        **Data Science & ML:**
        - 🐍 Python (Core Language)
        - 🐼 Pandas (Data Manipulation)
        - 🤖 Scikit-learn (Machine Learning)
        - 🔢 NumPy (Numerical Computing)
        """)
    
    st.markdown("---")
    st.info("👈 Use the navigation menu on the left to explore our data analysis and predictions!")

# EDA Page
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    
    # Generate sample data for demonstration
    @st.cache_data
    def load_airbnb_data():
        try:
            df = pd.read_csv('Airbnb_site_hotel new.csv')
            
            # Remove ID columns that are not useful for analysis
            columns_to_remove = ['id', 'host_id']
            df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
            
            # Clean the data
            # Convert price to numeric (remove commas if any)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Convert other numeric columns with proper handling of commas as decimal separators
            numeric_columns = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
                              'total reviewers number', 'host total listings count']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            
            # Fill missing values
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading Airbnb data: {e}")
            return pd.DataFrame()
    
    df = load_airbnb_data()
    
    st.subheader("📋 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns))
    with col3:
        st.metric("Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Visualizations
    st.subheader("📊 Data Visualizations")
    
    # Key insights about Airbnb data
    col1, col2, col3 = st.columns(3)
    with col1:
        avg_price = df['price'].mean()
        st.metric("Average Price", f"${avg_price:.2f}")
    with col2:
        total_listings = len(df)
        st.metric("Total Listings", f"{total_listings:,}")
    with col3:
        unique_cities = df['city'].nunique() if 'city' in df.columns else 0
        st.metric("Cities", unique_cities)
    
    # Price distribution
    st.subheader("💰 Price Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='price', title="Price Distribution", 
                          labels={'price': 'Price ($)', 'count': 'Number of Listings'})
        st.plotly_chart(fig, width='stretch')
        st.info("📊 **Insight:** Most Airbnb listings are priced between $50-200, with a right-skewed distribution indicating some premium properties at higher price points.")
    
    with col2:
        if 'room_type' in df.columns:
            fig = px.box(df, x='room_type', y='price', title="Price by Room Type")
            st.plotly_chart(fig, width='stretch')
            st.info("🏠 **Key Finding:** Entire homes command the highest prices, followed by private rooms, with shared rooms being the most affordable option.")
    
    # Room type analysis
    if 'room_type' in df.columns:
        st.subheader("🏠 Room Type Distribution")
        room_counts = df['room_type'].value_counts()
        fig = px.pie(values=room_counts.values, names=room_counts.index, 
                     title="Distribution of Room Types")
        st.plotly_chart(fig, width='stretch')
        st.success("📈 **Market Insight:** The distribution shows the market composition - understanding which property types dominate helps hosts and travelers make informed decisions.")
    
    # Geographic analysis
    if 'city' in df.columns:
        st.subheader("🌍 Geographic Distribution")
        city_counts = df['city'].value_counts().head(10)
        fig = px.bar(x=city_counts.values, y=city_counts.index, 
                     orientation='h', title="Top 10 Cities by Listings",
                     color=city_counts.values, 
                     color_continuous_scale='viridis',
                     labels={'x': 'Number of Listings', 'y': 'Cities'})
        st.plotly_chart(fig, width='stretch')
        st.info("🗺️ **Geographic Insight:** The concentration of listings reveals market hotspots and tourism patterns. Cities with higher listings typically indicate stronger demand and investment opportunities.")
    
    # Correlation heatmap for numeric features
    st.subheader("🔥 Correlation Matrix")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, 
                        text_auto=True, 
                        aspect="auto",
                        title="Numeric Features Correlation",
                        color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, width='stretch')
        st.warning("🔍 **Statistical Insight:** Strong correlations (darker colors) reveal relationships between features. High positive correlations suggest features move together, while negative correlations indicate inverse relationships.")
    
    # Feature Relationships Analysis
    st.subheader("🎯 Feature Relationships & Comparisons")
    
    # Price by different categories - easier to understand
    st.markdown("### 💰 **Price Analysis by Categories**")
    
    if 'room_type' in df.columns:
        # Average price by room type - Bar chart
        col1, col2 = st.columns(2)
        
        with col1:
            avg_price_by_room = df.groupby('room_type')['price'].mean().sort_values(ascending=False)
            fig = px.bar(x=avg_price_by_room.index, y=avg_price_by_room.values,
                        title="Average Price by Room Type",
                        labels={'x': 'Room Type', 'y': 'Average Price ($)'},
                        color=avg_price_by_room.values,
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
            st.info("🏠 **Clear Insight:** This shows which room types cost more on average. Entire homes are typically the most expensive.")
        
        with col2:
            # Count of listings by room type
            room_counts = df['room_type'].value_counts()
            fig = px.bar(x=room_counts.index, y=room_counts.values,
                        title="Number of Listings by Room Type",
                        labels={'x': 'Room Type', 'y': 'Number of Listings'},
                        color=room_counts.values,
                        color_continuous_scale='plasma')
            st.plotly_chart(fig, use_container_width=True)
            st.info("📊 **Market Share:** This shows which types of properties are most common in the market.")
    
    # Price ranges analysis
    st.markdown("### 💵 **Price Range Analysis**")
    col1, col2 = st.columns(2)
    
    with col1:
        # Create price ranges for better understanding
        df['price_range'] = pd.cut(df['price'], 
                                  bins=[0, 50, 100, 200, 500, float('inf')], 
                                  labels=['Under $50', '$50-100', '$100-200', '$200-500', 'Over $500'])
        price_range_counts = df['price_range'].value_counts().sort_index()
        
        fig = px.bar(x=price_range_counts.index, y=price_range_counts.values,
                    title="Distribution of Listings by Price Range",
                    labels={'x': 'Price Range', 'y': 'Number of Listings'},
                    color=price_range_counts.values,
                    color_continuous_scale='blues')
        st.plotly_chart(fig, use_container_width=True)
        st.success("💡 **Easy Understanding:** Most listings fall in the $50-200 range, making it the sweet spot for both hosts and guests.")
    
    with col2:
        # Accommodates vs average price
        if 'accommodates' in df.columns:
            avg_price_by_guests = df.groupby('accommodates')['price'].mean().head(10)
            fig = px.bar(x=avg_price_by_guests.index, y=avg_price_by_guests.values,
                        title="Average Price by Number of Guests",
                        labels={'x': 'Number of Guests', 'y': 'Average Price ($)'},
                        color=avg_price_by_guests.values,
                        color_continuous_scale='reds')
            st.plotly_chart(fig, use_container_width=True)
            st.info("👥 **Capacity Pricing:** Larger properties that accommodate more guests typically cost more per night.")
    
    # Interactive comparison tool
    st.markdown("### 🔍 **Compare Any Two Features**")
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select First Feature:", numeric_features, 
                                index=numeric_features.index('price') if 'price' in numeric_features else 0)
    with col2:
        y_feature = st.selectbox("Select Second Feature:", numeric_features, 
                                index=numeric_features.index('accommodates') if 'accommodates' in numeric_features else 1)
    
    # Create a cleaner scatter plot
    sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
    fig = px.scatter(sample_df, x=x_feature, y=y_feature, 
                     title=f"Relationship: {x_feature.replace('_', ' ').title()} vs {y_feature.replace('_', ' ').title()}",
                     opacity=0.7)
    fig.update_traces(marker=dict(size=6, color='steelblue'))
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Explain the relationship
    correlation = df[x_feature].corr(df[y_feature])
    if correlation > 0.5:
        relationship = "Strong Positive"
        explanation = f"As {x_feature.replace('_', ' ')} increases, {y_feature.replace('_', ' ')} tends to increase significantly."
    elif correlation > 0.2:
        relationship = "Moderate Positive"  
        explanation = f"As {x_feature.replace('_', ' ')} increases, {y_feature.replace('_', ' ')} tends to increase somewhat."
    elif correlation < -0.5:
        relationship = "Strong Negative"
        explanation = f"As {x_feature.replace('_', ' ')} increases, {y_feature.replace('_', ' ')} tends to decrease significantly."
    elif correlation < -0.2:
        relationship = "Moderate Negative"
        explanation = f"As {x_feature.replace('_', ' ')} increases, {y_feature.replace('_', ' ')} tends to decrease somewhat."
    else:
        relationship = "Weak/No"
        explanation = f"There's little to no clear relationship between {x_feature.replace('_', ' ')} and {y_feature.replace('_', ' ')}."
    
    st.warning(f"📈 **{relationship} Relationship** (Correlation: {correlation:.3f}): {explanation}")

# Prediction Page
elif page == "Prediction":
    st.title("🔮 Prediction Model")
    st.markdown("---")
    
    # Load Airbnb data for prediction
    @st.cache_data
    def load_airbnb_for_prediction():
        try:
            df = pd.read_csv('Airbnb_site_hotel new.csv')
            
            # Remove ID columns that are not useful for prediction
            columns_to_remove = ['id', 'host_id']
            df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])
            
            # Clean the data for prediction
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Convert numeric columns
            numeric_columns = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
                              'total reviewers number', 'host total listings count']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            
            # Remove rows with missing price (our target) and extreme outliers
            df = df.dropna(subset=['price'])
            df = df[df['price'] > 0]  # Remove zero or negative prices
            
            # Fill other missing values with median for better model performance
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
            
            return df
            
        except Exception as e:
            st.error(f"Error loading Airbnb data for prediction: {e}")
            return pd.DataFrame()
    
    df = load_airbnb_for_prediction()
    
    if len(df) == 0:
        st.error("No data available for prediction.")
        st.stop()
    
    st.subheader("🎯 Airbnb Price Prediction Model")
    
    # Model Information
    st.info("🤖 **Model:** Linear Regression | **Algorithm:** Ordinary Least Squares | **Purpose:** Predicting Airbnb listing prices based on property features")
    
    # Select features for prediction
    feature_columns = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
                      'total reviewers number', 'host total listings count']
    
    # Filter available columns
    available_features = [col for col in feature_columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
    
    if len(available_features) == 0:
        st.error("No suitable numeric features found for prediction.")
        st.stop()
    
    # Prepare data with better preprocessing
    X = df[available_features].copy()
    y = df['price'].copy()
    
    # Remove outliers using IQR method for better model performance
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    mask = (y >= lower_bound) & (y <= upper_bound) & (y > 0)
    X = X[mask]
    y = y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    st.success(f"✅ **Model trained on {len(X)} listings** after removing outliers and missing values for optimal performance.")
    
    # Model performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", f"{r2:.3f}")
    with col2:
        st.metric("MSE", f"{mse:.3f}")
    with col3:
        st.metric("RMSE", f"{np.sqrt(mse):.3f}")
    
    # Actual vs Predicted plot
    st.subheader("📈 Model Performance")
    
    # Explanation of what we're predicting
    st.info(f"""
    🎯 **What We're Predicting:** Airbnb listing **PRICE** (in dollars per night)
    
    📊 **Features Used for Prediction:** {', '.join([f.replace('_', ' ').title() for f in available_features])}
    
    🎪 **How It Works:** 
    - **Actual Values (X-axis):** Real prices from Airbnb listings in our dataset
    - **Predicted Values (Y-axis):** What our AI model guessed the price should be
    - **Goal:** Points closer to the red dashed line = better predictions
    - **Perfect Prediction:** If our model was 100% accurate, all dots would be on the red line
    """)
    
    fig = px.scatter(x=y_test, y=y_pred, 
                     title="Actual vs Predicted Prices ($)",
                     labels={'x': 'Actual Price ($)', 'y': 'Predicted Price ($)'})
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', name='Perfect Prediction Line',
                            line=dict(dash='dash', color='red', width=3)))
    
    st.plotly_chart(fig, width='stretch')
    
    # Performance interpretation
    if r2 > 0.7:
        performance = "Excellent"
        color = "success"
    elif r2 > 0.5:
        performance = "Good" 
        color = "info"
    elif r2 > 0.3:
        performance = "Fair"
        color = "warning"
    else:
        performance = "Needs Improvement"
        color = "error"
    
    if color == "success":
        st.success(f"🎉 **{performance} Model Performance!** Our AI can predict Airbnb prices with {r2*100:.1f}% accuracy using property features.")
    elif color == "info":
        st.info(f"✅ **{performance} Model Performance!** Our AI can predict Airbnb prices with {r2*100:.1f}% accuracy using property features.")
    elif color == "warning":
        st.warning(f"⚠️ **{performance} Model Performance.** Our AI can predict Airbnb prices with {r2*100:.1f}% accuracy using property features.")
    else:
        st.error(f"❌ **{performance}** - Our AI can predict Airbnb prices with {r2*100:.1f}% accuracy using property features.")
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': available_features,
        'Importance': np.abs(model.coef_)
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance, x='Feature', y='Importance',
                 title="Feature Importance for Price Prediction")
    st.plotly_chart(fig, width='stretch')
    
    # Interactive prediction
    st.subheader("🔮 Predict Airbnb Price")
    st.write("Enter the property details below to get an estimated price:")
    
    # Create number inputs based on available features
    user_input = {}
    cols = st.columns(2)
    
    # Define user-friendly labels and reasonable defaults
    feature_info = {
        'accommodates': {'label': 'Number of Guests', 'default': 2, 'min': 1, 'max': 20},
        'bathrooms': {'label': 'Number of Bathrooms', 'default': 1, 'min': 1, 'max': 10},
        'bedrooms': {'label': 'Number of Bedrooms', 'default': 1, 'min': 0, 'max': 10},
        'beds': {'label': 'Number of Beds', 'default': 1, 'min': 1, 'max': 20},
        'total reviewers number': {'label': 'Total Reviews', 'default': 5, 'min': 0, 'max': 500},
        'host total listings count': {'label': 'Host Total Listings', 'default': 1, 'min': 1, 'max': 100}
    }
    
    for i, feature in enumerate(available_features):
        col_idx = i % 2
        with cols[col_idx]:
            info = feature_info.get(feature, {'label': feature.replace('_', ' ').title(), 'default': 1, 'min': 0, 'max': 100})
            
            if 'step' in info:
                user_input[feature] = st.number_input(
                    info['label'],
                    min_value=info['min'],
                    max_value=info['max'],
                    value=info['default'],
                    step=info['step']
                )
            else:
                user_input[feature] = st.number_input(
                    info['label'],
                    min_value=info['min'],
                    max_value=info['max'],
                    value=info['default'],
                    step=1
                )
    
    # Make prediction
    input_array = np.array([[user_input[feature] for feature in available_features]])
    predicted_price = model.predict(input_array)[0]
    
    st.success(f"💰 Predicted Airbnb Price: ${predicted_price:.2f}")
    
    # Show input values
    st.subheader("📊 Input Summary")
    input_df = pd.DataFrame({
        'Feature': [feature.replace('_', ' ').title() for feature in available_features],
        'Value': [user_input[feature] for feature in available_features]
    })
    st.dataframe(input_df, width='stretch')

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Data Analysis Dashboard © 2025</p>
    </div>
    """, 
    unsafe_allow_html=True
)
