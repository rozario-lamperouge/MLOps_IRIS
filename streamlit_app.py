"""
MLOps Iris Classifier - Streamlit App
Train models, track experiments, make predictions
"""
import streamlit as st
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="MLOps Iris Classifier",
    page_icon="🌸",
    layout="wide"
)

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("iris-classification")

# Load iris data
iris = load_iris()
species_names = ['Setosa', 'Versicolor', 'Virginica']

# Ensure model directory exists
Path("models").mkdir(exist_ok=True)


def train_model(n_estimators, max_depth, min_samples_split, test_size):
    """Train model and log to MLflow"""
    
    # Split data
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"rf_n{n_estimators}_d{max_depth}"):
        
        # Log parameters
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "test_size": test_size
        }
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        joblib.dump(model, "models/iris_model.pkl")
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return model, accuracy, f1, cm, mlflow.active_run().info.run_id


def load_model():
    """Load the latest trained model"""
    model_path = Path("models/iris_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    return None


# Sidebar navigation
st.sidebar.title("🌸 MLOps Iris Classifier")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 About", "🧪 Train Model", "🔮 Make Predictions", "📊 MLflow Dashboard"]
)

# ==================== ABOUT PAGE ====================
if page == "🏠 About":
    st.title("🌸 MLOps Iris Classification")
    st.markdown("### A Complete MLOps Workflow Demo")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 What is this?
        
        This app demonstrates a complete **MLOps workflow**:
        
        1. **Train** machine learning models with different hyperparameters
        2. **Track** experiments automatically with MLflow
        3. **Deploy** models for real-time predictions
        4. **Monitor** model performance
        
        ## 🌺 The Dataset: Iris Flowers
        
        The famous Iris dataset contains measurements of 150 iris flowers from 3 species:
        - **Setosa** 🌼
        - **Versicolor** 🌸  
        - **Virginica** 🌺
        
        **Features measured:**
        - Sepal Length (cm)
        - Sepal Width (cm)
        - Petal Length (cm)
        - Petal Width (cm)
        
        ## 🤖 The Model
        
        We use a **Random Forest Classifier**:
        - An ensemble of decision trees
        - Robust and accurate
        - Easy to interpret
        - Works well with small datasets
        
        ## 🔄 MLOps Workflow
        
        ```
        Train Model → Log to MLflow → Compare Experiments → Deploy Best Model
        ```
        """)
    
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", 
                 caption="Iris Versicolor")
        
        st.metric("Total Samples", "150")
        st.metric("Features", "4")
        st.metric("Classes", "3")
        st.metric("Model Type", "Random Forest")
    
    st.markdown("---")
    st.info("👈 Use the sidebar to navigate: Train models, make predictions, or view MLflow experiments!")

# ==================== TRAIN MODEL PAGE ====================
elif page == "🧪 Train Model":
    st.title("🧪 Train Model & Track Experiments")
    st.markdown("### Adjust hyperparameters and train your model")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### 🎛️ Hyperparameters")
        
        n_estimators = st.slider(
            "Number of Trees (n_estimators)",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="More trees = more accurate but slower"
        )
        
        max_depth = st.slider(
            "Maximum Tree Depth (max_depth)",
            min_value=1,
            max_value=20,
            value=5,
            help="Deeper trees = more complex model"
        )
        
        min_samples_split = st.slider(
            "Minimum Samples to Split (min_samples_split)",
            min_value=2,
            max_value=20,
            value=2,
            help="Higher = more conservative splitting"
        )
        
        test_size = st.slider(
            "Test Set Size (%)",
            min_value=10,
            max_value=50,
            value=20,
            help="Percentage of data for testing"
        ) / 100
        
        train_button = st.button("🚀 Train Model", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Training Results")
        results_placeholder = st.empty()
        
    if train_button:
        with st.spinner("🔄 Training model..."):
            # Train model
            model, accuracy, f1, cm, run_id = train_model(
                n_estimators, max_depth, min_samples_split, test_size
            )
            
            # Show results
            with results_placeholder.container():
                st.success("✅ Model trained successfully!")
                
                # Metrics
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("Accuracy", f"{accuracy:.4f}")
                metric_col2.metric("F1 Score", f"{f1:.4f}")
                
                # Confusion Matrix
                st.markdown("##### Confusion Matrix")
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=species_names,
                    y=species_names,
                    text_auto=True,
                    color_continuous_scale="Blues"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.info(f"📝 **MLflow Run ID:** `{run_id}`")
                st.info("🔍 Open MLflow UI to compare experiments: `mlflow ui`")
    
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips for Training
    
    - **Start with defaults** and see the baseline performance
    - **Increase n_estimators** for better accuracy (but slower training)
    - **Increase max_depth** if underfitting (low accuracy)
    - **Increase min_samples_split** if overfitting (train accuracy >> test accuracy)
    - **Compare results** in MLflow to find the best model!
    """)

# ==================== PREDICTION PAGE ====================
elif page == "🔮 Make Predictions":
    st.title("🔮 Make Predictions")
    st.markdown("### Use the trained model to predict iris species")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("❌ No model found! Please train a model first.")
        st.info("👈 Go to 'Train Model' page to train your first model")
    else:
        st.success("✅ Model loaded successfully!")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📏 Enter Flower Measurements")
            
            sepal_length = st.number_input(
                "Sepal Length (cm)",
                min_value=0.0,
                max_value=10.0,
                value=5.1,
                step=0.1
            )
            
            sepal_width = st.number_input(
                "Sepal Width (cm)",
                min_value=0.0,
                max_value=10.0,
                value=3.5,
                step=0.1
            )
            
            petal_length = st.number_input(
                "Petal Length (cm)",
                min_value=0.0,
                max_value=10.0,
                value=1.4,
                step=0.1
            )
            
            petal_width = st.number_input(
                "Petal Width (cm)",
                min_value=0.0,
                max_value=10.0,
                value=0.2,
                step=0.1
            )
            
            predict_button = st.button("🔮 Predict Species", type="primary", use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Prediction Results")
            
            if predict_button:
                # Prepare input
                features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
                
                # Predict
                prediction = model.predict(features)[0]
                probabilities = model.predict_proba(features)[0]
                
                # Show prediction
                st.markdown(f"### Predicted Species: **{species_names[prediction]}**")
                st.markdown(f"**Confidence:** {probabilities[prediction]*100:.2f}%")
                
                # Show probabilities
                st.markdown("##### Class Probabilities")
                prob_df = pd.DataFrame({
                    'Species': species_names,
                    'Probability': probabilities
                })
                
                fig = px.bar(
                    prob_df,
                    x='Species',
                    y='Probability',
                    color='Probability',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show input summary
                st.markdown("##### Input Summary")
                input_df = pd.DataFrame({
                    'Feature': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
                    'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
                })
                st.dataframe(input_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("""
        ### 📝 Example Measurements
        
        Try these typical measurements for each species:
        
        | Species | Sepal Length | Sepal Width | Petal Length | Petal Width |
        |---------|--------------|-------------|--------------|-------------|
        | **Setosa** | 5.1 | 3.5 | 1.4 | 0.2 |
        | **Versicolor** | 5.9 | 3.0 | 4.2 | 1.5 |
        | **Virginica** | 6.5 | 3.0 | 5.5 | 2.0 |
        """)

# ==================== MLFLOW DASHBOARD PAGE ====================
elif page == "📊 MLflow Dashboard":
    st.title("📊 MLflow Experiment Tracking")
    st.markdown("### View and compare your experiments")
    
    st.markdown("""
    ## 🚀 How to Access MLflow UI
    
    MLflow provides a powerful web interface to track and compare experiments.
    
    ### Step 1: Start MLflow UI
    
    Open a **new terminal** and run:
    
    ```bash
    mlflow ui
    ```
    
    ### Step 2: Open in Browser
    
    Navigate to: **http://localhost:5000**
    
    ### What You'll See:
    
    - 📊 **All experiments** in a table
    - 📈 **Metrics comparison** across runs
    - 🎛️ **Parameter values** for each run
    - 📁 **Logged models** ready for deployment
    - 📉 **Visualization** of metrics over time
    
    ---
    
    ## 💡 Tips for Using MLflow
    
    1. **Compare runs** - Select multiple runs and click "Compare"
    2. **Sort by metrics** - Click column headers to find best models
    3. **Filter experiments** - Use search to find specific runs
    4. **Download models** - Get artifacts from any run
    5. **Share results** - Export run data or share screenshots
    
    ---
    
    ## 🎯 What to Look For
    
    When comparing experiments, consider:
    
    - ✅ **Accuracy** - Higher is better
    - ✅ **F1 Score** - Balance of precision and recall
    - ⚡ **Training time** - Faster is better for iteration
    - 📊 **Overfitting** - Train vs test performance gap
    
    """)
    
    st.info("💻 **Local Training Session**: MLflow UI is designed to run locally during training. Keep it open alongside this Streamlit app!")
    
    st.warning("🌐 **Deployed Version**: MLflow tracking is not available in the deployed app. Use it during local development!")
    
    # Show example of what MLflow looks like
    st.markdown("### 📸 MLflow UI Preview")
    st.markdown("This is what you'll see in MLflow:")
    
    st.code("""
    Run ID              | Metrics       | Parameters              | Start Time
    ------------------- | ------------- | ----------------------- | -------------------
    abc123              | accuracy: 0.97| n_estimators: 100       | 2024-01-15 10:30:00
                        | f1_score: 0.96| max_depth: 5            |
    
    def456              | accuracy: 0.95| n_estimators: 200       | 2024-01-15 10:35:00
                        | f1_score: 0.94| max_depth: 3            |
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Resources")
st.sidebar.markdown("""
- [MLflow Docs](https://mlflow.org/docs/latest/index.html)
- [Streamlit Docs](https://docs.streamlit.io)
- [Scikit-learn Docs](https://scikit-learn.org)
""")
st.sidebar.info("Made with ❤️ for MLOps Training")
