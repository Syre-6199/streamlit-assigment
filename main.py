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
    st.title("📋 About Us")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://via.placeholder.com/300x200?text=Our+Team", caption="Our Team")
    
    with col2:
        st.subheader("Welcome to Our Data Analysis Platform")
        st.write("""
        We are a team of data scientists and analysts passionate about turning data into insights.
        Our mission is to make data analysis accessible and understandable for everyone.
        """)
        
        st.subheader("Our Expertise")
        st.write("• Data Analysis & Visualization")
        st.write("• Machine Learning & Predictive Modeling")
        st.write("• Statistical Analysis")
        st.write("• Business Intelligence")
    
    st.markdown("---")
    st.subheader("Contact Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📧 Email: contact@dataanalysis.com")
    with col2:
        st.info("📞 Phone: +1 (555) 123-4567")
    with col3:
        st.info("🌐 Website: www.dataanalysis.com")

# Introduction Page
elif page == "Introduction":
    st.title("🚀 Introduction")
    st.markdown("---")
    
    st.subheader("Welcome to Our Data Analysis Dashboard")
    st.write("""
    This interactive dashboard provides comprehensive data analysis capabilities including:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Exploratory Data Analysis (EDA)
        - Interactive visualizations
        - Statistical summaries
        - Data distribution analysis
        - Correlation analysis
        """)
        
        st.markdown("""
        ### 🔮 Predictive Modeling
        - Machine learning algorithms
        - Model performance metrics
        - Interactive predictions
        - Feature importance analysis
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Key Features
        - User-friendly interface
        - Real-time data processing
        - Customizable visualizations
        - Export capabilities
        """)
        
        st.markdown("""
        ### 📈 Benefits
        - Make data-driven decisions
        - Identify trends and patterns
        - Predict future outcomes
        - Improve business performance
        """)
    
    st.markdown("---")
    st.info("👈 Use the navigation menu on the left to explore different sections of our platform.")

# EDA Page
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis")
    st.markdown("---")
    
    # Generate sample data for demonstration
    @st.cache_data
    def generate_sample_data():
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Age': np.random.normal(35, 10, n_samples).astype(int),
            'Income': np.random.normal(50000, 15000, n_samples),
            'Education_Years': np.random.normal(16, 3, n_samples),
            'Experience': np.random.normal(10, 5, n_samples),
            'Score': np.random.normal(75, 15, n_samples)
        }
        
        # Add some correlation
        data['Income'] = data['Income'] + data['Education_Years'] * 2000 + np.random.normal(0, 5000, n_samples)
        data['Score'] = data['Score'] + data['Experience'] * 2 + np.random.normal(0, 10, n_samples)
        
        return pd.DataFrame(data)
    
    df = generate_sample_data()
    
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
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        feature = st.selectbox("Select feature for distribution:", df.columns)
        fig = px.histogram(df, x=feature, title=f"Distribution of {feature}")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(df, y=feature, title=f"Box Plot of {feature}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("🔥 Correlation Matrix")
    corr_matrix = df.corr()
    fig = px.imshow(corr_matrix, 
                    text_auto=True, 
                    aspect="auto",
                    title="Feature Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot
    st.subheader("🎯 Feature Relationships")
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox("Select X-axis:", df.columns, index=0)
    with col2:
        y_feature = st.selectbox("Select Y-axis:", df.columns, index=1)
    
    fig = px.scatter(df, x=x_feature, y=y_feature, 
                     title=f"{x_feature} vs {y_feature}")
    st.plotly_chart(fig, use_container_width=True)

# Prediction Page
elif page == "Prediction":
    st.title("🔮 Prediction Model")
    st.markdown("---")
    
    # Generate sample data
    @st.cache_data
    def generate_prediction_data():
        np.random.seed(42)
        n_samples = 1000
        
        X = np.random.randn(n_samples, 4)
        y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + 0.5*X[:, 3] + np.random.randn(n_samples) * 0.5
        
        feature_names = ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4']
        df = pd.DataFrame(X, columns=feature_names)
        df['Target'] = y
        
        return df
    
    df = generate_prediction_data()
    
    st.subheader("🎯 Model Training")
    
    # Model training
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(model.coef_)
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(feature_importance, x='Feature', y='Importance',
                 title="Feature Importance (Absolute Coefficients)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive prediction
    st.subheader("🔮 Make a Prediction")
    st.write("Adjust the feature values below to make a prediction:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        feature_1 = st.slider("Feature 1", -3.0, 3.0, 0.0, 0.1)
        feature_2 = st.slider("Feature 2", -3.0, 3.0, 0.0, 0.1)
    
    with col2:
        feature_3 = st.slider("Feature 3", -3.0, 3.0, 0.0, 0.1)
        feature_4 = st.slider("Feature 4", -3.0, 3.0, 0.0, 0.1)
    
    # Make prediction
    input_features = np.array([[feature_1, feature_2, feature_3, feature_4]])
    prediction = model.predict(input_features)[0]
    
    st.success(f"🎯 Predicted Value: {prediction:.3f}")
    
    # Show input values
    st.subheader("📊 Input Summary")
    input_df = pd.DataFrame({
        'Feature': ['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4'],
        'Value': [feature_1, feature_2, feature_3, feature_4]
    })
    st.dataframe(input_df, use_container_width=True)

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