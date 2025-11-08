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
    
    with col2:
        if 'room_type' in df.columns:
            fig = px.box(df, x='room_type', y='price', title="Price by Room Type")
            st.plotly_chart(fig, width='stretch')
    
    # Room type analysis
    if 'room_type' in df.columns:
        st.subheader("🏠 Room Type Distribution")
        room_counts = df['room_type'].value_counts()
        fig = px.pie(values=room_counts.values, names=room_counts.index, 
                     title="Distribution of Room Types")
        st.plotly_chart(fig, width='stretch')
    
    # Geographic analysis
    if 'city' in df.columns:
        st.subheader("🌍 Geographic Distribution")
        city_counts = df['city'].value_counts().head(10)
        fig = px.bar(x=city_counts.values, y=city_counts.index, 
                     orientation='h', title="Top 10 Cities by Listings")
        st.plotly_chart(fig, width='stretch')
    
    # Correlation heatmap for numeric features
    st.subheader("🔥 Correlation Matrix")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr()
        fig = px.imshow(corr_matrix, 
                        text_auto=True, 
                        aspect="auto",
                        title="Numeric Features Correlation")
        st.plotly_chart(fig, width='stretch')
    
    # Interactive scatter plot
    st.subheader("🎯 Feature Relationships")
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select X-axis:", numeric_features, 
                                index=numeric_features.index('price') if 'price' in numeric_features else 0)
    with col2:
        y_feature = st.selectbox("Select Y-axis:", numeric_features, 
                                index=numeric_features.index('accommodates') if 'accommodates' in numeric_features else 1)
    
    if 'room_type' in df.columns:
        fig = px.scatter(df, x=x_feature, y=y_feature, color='room_type',
                         title=f"{x_feature} vs {y_feature}")
    else:
        fig = px.scatter(df, x=x_feature, y=y_feature,
                         title=f"{x_feature} vs {y_feature}")
    st.plotly_chart(fig, width='stretch')

# Prediction Page
elif page == "Prediction":
    st.title("🔮 Prediction Model")
    st.markdown("---")
    
    # Load Airbnb data for prediction
    @st.cache_data
    def load_airbnb_for_prediction():
        try:
            df = pd.read_csv('Airbnb_site_hotel new.csv')
            
            # Clean the data for prediction
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            
            # Convert numeric columns
            numeric_columns = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
                              'total reviewers number', 'host total listings count']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            
            # Remove rows with missing price (our target)
            df = df.dropna(subset=['price'])
            
            # Fill other missing values
            df = df.fillna(0)
            
            return df
            
        except Exception as e:
            st.error(f"Error loading Airbnb data for prediction: {e}")
            return pd.DataFrame()
    
    df = load_airbnb_for_prediction()
    
    if len(df) == 0:
        st.error("No data available for prediction.")
        st.stop()
    
    st.subheader("🎯 Airbnb Price Prediction Model")
    
    # Select features for prediction
    feature_columns = ['accommodates', 'bathrooms', 'bedrooms', 'beds', 
                      'total reviewers number', 'host total listings count']
    
    # Filter available columns
    available_features = [col for col in feature_columns if col in df.columns and df[col].dtype in ['int64', 'float64']]
    
    if len(available_features) == 0:
        st.error("No suitable numeric features found for prediction.")
        st.stop()
    
    # Prepare data
    X = df[available_features]
    y = df['price']
    
    # Remove outliers (prices above 95th percentile)
    price_threshold = y.quantile(0.95)
    mask = y <= price_threshold
    X = X[mask]
    y = y[mask]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
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
    fig = px.scatter(x=y_test, y=y_pred, 
                     title="Actual vs Predicted Values",
                     labels={'x': 'Actual Values', 'y': 'Predicted Values'})
    
    # Add perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                            mode='lines', name='Perfect Prediction',
                            line=dict(dash='dash', color='red')))
    
    st.plotly_chart(fig, width='stretch')
    
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
    st.write("Adjust the values below to predict the Airbnb price:")
    
    # Create input sliders based on available features
    user_input = {}
    cols = st.columns(2)
    
    for i, feature in enumerate(available_features):
        col_idx = i % 2
        with cols[col_idx]:
            min_val = float(X[feature].min())
            max_val = float(X[feature].max())
            mean_val = float(X[feature].mean())
            
            user_input[feature] = st.slider(
                f"{feature.replace('_', ' ').title()}", 
                min_val, max_val, mean_val, 
                step=(max_val - min_val) / 100
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
