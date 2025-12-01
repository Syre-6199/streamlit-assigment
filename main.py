import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline

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
            columns_to_remove = ['id', 'host_id', 'host_name', 'listingh number', 'listing number', 'listing_number']
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
            
            # Fill missing values for numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            df[numeric_cols] = df[numeric_cols].fillna(0)

            # Map numeric room_type codes to readable labels for EDA clarity
            if 'room_type' in df.columns:
                # try to coerce to integer codes then map to labels
                room_codes = pd.to_numeric(df['room_type'], errors='coerce').fillna(0).astype(int)
                # Swap codes 1 and 3: code 1 -> Shared room, code 3 -> Entire home/apt
                room_map = {
                    1: 'Shared room',
                    2: 'Private room',
                    3: 'Entire home/apt',
                    4: 'Hotel room'
                }
                df['room_type'] = room_codes.map(room_map).fillna('Other')
            
            return df
            
        except Exception as e:
            st.error(f"Error loading Airbnb data: {e}")
            return pd.DataFrame()
    
    df = load_airbnb_data()

    # Show room type legend if codes were mapped (keep consistent with mapping above)
    if 'room_type' in df.columns:
        st.markdown("**Room type labels:** 1 → Shared room, 2 → Private room, 3 → Entire home/apt, 4 → Hotel room")
    
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
    
    # Price distribution (simpler, more readable)
    st.subheader("💰 Price Analysis")
    col1, col2 = st.columns(2)

    with col1:
        # Create clear price range buckets and show counts
        df['price_range'] = pd.cut(df['price'], 
                                  bins=[0, 50, 100, 200, 500, float('inf')], 
                                  labels=['Under $50', '$50-100', '$100-200', '$200-500', 'Over $500'])
        price_range_counts = df['price_range'].value_counts().sort_index()

        fig = px.bar(x=price_range_counts.index, y=price_range_counts.values,
                     title="Listings by Price Range",
                     labels={'x': 'Price Range', 'y': 'Number of Listings'},
                     color=price_range_counts.values,
                     color_continuous_scale='blues')
        st.plotly_chart(fig, use_container_width=True)
        st.info("Most listings fall in the $50-200 range. Use this to quickly see market concentration by price bands.")

    with col2:
        # Show average price per room type as a simple bar chart for easy comparison
        if 'room_type' in df.columns:
            avg_price_by_room = df.groupby('room_type')['price'].mean().sort_values(ascending=False)
            fig = px.bar(x=avg_price_by_room.index, y=avg_price_by_room.values,
                        title="Average Price by Room Type",
                        labels={'x': 'Room Type', 'y': 'Average Price ($)'},
                        color=avg_price_by_room.values,
                        color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
            st.info("Average prices by room type — easier to compare than boxplots for quick summaries.")
    
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
        # Show count of listings by room type (avoid duplicate avg price chart)
        col1, col2 = st.columns(2)
        
        with col1:
            # Count of listings by room type
            room_counts = df['room_type'].value_counts()
            fig = px.bar(x=room_counts.index, y=room_counts.values,
                        title="Number of Listings by Room Type",
                        labels={'x': 'Room Type', 'y': 'Number of Listings'},
                        color=room_counts.values,
                        color_continuous_scale='plasma')
            st.plotly_chart(fig, use_container_width=True)
            st.info("📊 **Market Share:** This shows which types of properties are most common in the market.")
        
        with col2:
            # Median price by room type for comparison
            median_price_by_room = df.groupby('room_type')['price'].median().sort_values(ascending=False)
            fig = px.bar(x=median_price_by_room.index, y=median_price_by_room.values,
                        title="Median Price by Room Type",
                        labels={'x': 'Room Type', 'y': 'Median Price ($)'},
                        color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig, use_container_width=True)
            st.info("💰 **Median Pricing:** Shows the middle-point price for each room type, less affected by outliers than average.")
    
    # Price ranges analysis
    st.markdown("### 💵 **Price Range Analysis**")
    col1, col2 = st.columns(2)
    
    with col1:
        # Use the price_range already created above
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
    
    # Interactive comparison tool with histograms
    st.markdown("### 🔍 **Compare Any Two Features**")
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_feature = st.selectbox("Select First Feature:", numeric_features, 
                                index=numeric_features.index('price') if 'price' in numeric_features else 0)
    with col2:
        y_feature = st.selectbox("Select Second Feature:", numeric_features, 
                                index=numeric_features.index('accommodates') if 'accommodates' in numeric_features else 1)
    
    # Create individual histograms for each feature
    st.markdown(f"#### Distribution Analysis")
    hist_col1, hist_col2 = st.columns(2)
    
    with hist_col1:
        # Histogram for first feature
        fig_x = px.histogram(df, x=x_feature, nbins=30,
                            title=f"Distribution of {x_feature.replace('_', ' ').title()}",
                            labels={x_feature: x_feature.replace('_', ' ').title()},
                            color_discrete_sequence=['#1f77b4'])
        fig_x.update_layout(showlegend=False)
        st.plotly_chart(fig_x, use_container_width=True)
    
    with hist_col2:
        # Histogram for second feature
        fig_y = px.histogram(df, x=y_feature, nbins=30,
                            title=f"Distribution of {y_feature.replace('_', ' ').title()}",
                            labels={y_feature: y_feature.replace('_', ' ').title()},
                            color_discrete_sequence=['#ff7f0e'])
        fig_y.update_layout(showlegend=False)
        st.plotly_chart(fig_y, use_container_width=True)
    
    # Create 2D relationship analysis with options: density heatmap, sampled scatter + contour, or sampled scatter
    st.markdown(f"#### Relationship Analysis")
    vis_option = st.selectbox("Choose relationship view:",
                              ["Density heatmap (default)", "Sampled scatter + contour", "Sampled scatter (with marginals)"],
                              index=0)

    # Controls for sampling and log-scaling
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        sample_frac = st.slider("Scatter sample fraction", min_value=0.01, max_value=0.5, value=0.05, step=0.01,
                                help="Fraction of rows to plot as points to avoid overplotting")
    with col_b:
        log_x = st.checkbox("Log scale X axis", value=False)
    with col_c:
        log_y = st.checkbox("Log scale Y axis", value=False)

    # Build and render selected visualization
    if vis_option == "Density heatmap (default)":
        fig = px.density_heatmap(df, x=x_feature, y=y_feature, nbinsx=30, nbinsy=30,
                                 title=f"{x_feature.replace('_', ' ').title()} vs {y_feature.replace('_', ' ').title()} - Density Heatmap",
                                 labels={x_feature: x_feature.replace('_', ' ').title(),
                                         y_feature: y_feature.replace('_', ' ').title()},
                                 color_continuous_scale='Blues')
        if log_x:
            fig.update_xaxes(type='log')
        if log_y:
            fig.update_yaxes(type='log')
        st.plotly_chart(fig, use_container_width=True)

    elif vis_option == "Sampled scatter + contour":
        # density contour (smooth) + sampled scatter overlay
        contour = px.density_contour(df, x=x_feature, y=y_feature, title=f"{x_feature.replace('_', ' ').title()} vs {y_feature.replace('_', ' ').title()} - Contour + Sampled Points",
                                    labels={x_feature: x_feature.replace('_', ' ').title(),
                                            y_feature: y_feature.replace('_', ' ').title()},
                                    color_continuous_scale='Blues')
        # sample points to avoid overplotting
        try:
            sample_df = df.sample(frac=sample_frac)
        except Exception:
            sample_df = df.copy()

        # color by room_type if available, otherwise no color
        if 'room_type' in df.columns:
            scatter = px.scatter(sample_df, x=x_feature, y=y_feature, color='room_type', opacity=0.6,
                                 labels={x_feature: x_feature.replace('_', ' ').title(), y_feature: y_feature.replace('_', ' ').title()})
        else:
            scatter = px.scatter(sample_df, x=x_feature, y=y_feature, opacity=0.6,
                                 labels={x_feature: x_feature.replace('_', ' ').title(), y_feature: y_feature.replace('_', ' ').title()})

        # overlay scatter traces onto contour figure
        for trace in scatter.data:
            contour.add_trace(trace)

        if log_x:
            contour.update_xaxes(type='log')
        if log_y:
            contour.update_yaxes(type='log')

        st.plotly_chart(contour, use_container_width=True)

    else:
        # Sampled scatter with marginals for context
        try:
            sample_df = df.sample(frac=sample_frac)
        except Exception:
            sample_df = df.copy()

        if 'room_type' in df.columns:
            fig = px.scatter(sample_df, x=x_feature, y=y_feature, color='room_type', opacity=0.6,
                             marginal_x='histogram', marginal_y='histogram',
                             title=f"{x_feature.replace('_', ' ').title()} vs {y_feature.replace('_', ' ').title()} - Sampled Scatter with Marginals",
                             labels={x_feature: x_feature.replace('_', ' ').title(), y_feature: y_feature.replace('_', ' ').title()})
        else:
            fig = px.scatter(sample_df, x=x_feature, y=y_feature, opacity=0.6,
                             marginal_x='histogram', marginal_y='histogram',
                             title=f"{x_feature.replace('_', ' ').title()} vs {y_feature.replace('_', ' ').title()} - Sampled Scatter with Marginals",
                             labels={x_feature: x_feature.replace('_', ' ').title(), y_feature: y_feature.replace('_', ' ').title()})

        if log_x:
            fig.update_xaxes(type='log')
        if log_y:
            fig.update_yaxes(type='log')

        st.plotly_chart(fig, use_container_width=True)

    # Show correlation statistics
    correlation = df[x_feature].corr(df[y_feature])
    
    if correlation > 0.5:
        relationship = "🔵 Strong Positive"
        emoji = "📈"
    elif correlation > 0.2:
        relationship = "🟢 Moderate Positive"
        emoji = "📊"
    elif correlation < -0.5:
        relationship = "🔴 Strong Negative"
        emoji = "📉"
    elif correlation < -0.2:
        relationship = "🟠 Moderate Negative"
        emoji = "📊"
    else:
        relationship = "⚪ Weak/No Relationship"
        emoji = "➡️"
    
    st.info(f"{emoji} **Correlation: {correlation:.3f}** — {relationship}")

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
            columns_to_remove = ['id', 'host_id', 'host_name', 'listingh number', 'listing number', 'listing_number']
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
    # Improve model: log-transform target, polynomial features + Ridge regression
    # log1p helps stabilize skewed price distribution
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    pipeline = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        StandardScaler(),
        Ridge(alpha=1.0)
    )

    pipeline.fit(X_train, y_train_log)

    # Predict on log scale then invert with expm1
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    
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
    
    # Feature importance (approx): fit a simple linear model on the log-target
    st.subheader("🎯 Feature Importance (approx.)")
    try:
        simple_lin = LinearRegression()
        simple_lin.fit(X_train, y_train_log)
        feature_importance = pd.DataFrame({
            'Feature': X_train.columns,
            'Importance': np.abs(simple_lin.coef_)
        }).sort_values('Importance', ascending=False)

        fig = px.bar(feature_importance, x='Feature', y='Importance',
                     title="Approximate Feature Importance (absolute coeffs on log-target)")
        st.plotly_chart(fig, width='stretch')
    except Exception:
        st.info("Feature importance could not be computed.")
    
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
    # Predict on log scale then invert
    predicted_price_log = pipeline.predict(input_array)[0]
    predicted_price = np.expm1(predicted_price_log)
    
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
