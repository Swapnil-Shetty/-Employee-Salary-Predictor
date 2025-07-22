# Employee Salary Predictor

A comprehensive machine learning project that predicts employee salary brackets (<=50K or >50K) based on demographic and employment characteristics.

## 🎯 Project Overview

This project implements a complete machine learning pipeline from data preprocessing to web deployment, designed to predict salary brackets using census data. The application provides insights into factors influencing income levels and serves as a practical tool for HR professionals and job seekers.

## 🚀 Features

- **Complete ML Pipeline**: Data loading, preprocessing, model training, and evaluation
- **Multiple Algorithm Comparison**: Tests various classification algorithms to find the best performer
- **Interactive Web Interface**: Beautiful Streamlit app for real-time predictions
- **Feature Engineering**: Intelligent data transformation and feature creation
- **Model Interpretability**: Feature importance analysis and prediction explanations
- **Production Ready**: Robust error handling and scalable architecture

## 📋 Requirements

### Python Libraries
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
joblib>=1.0.0
streamlit>=1.28.0
plotly>=5.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

### Installation
```bash
pip install -r requirements.txt
```

## 🗂️ Project Structure

```
salary-predictor/
├── 1_setup_data.py              # Data loading and initial inspection
├── 2_data_preprocessing.py      # Data cleaning and feature engineering
├── 3_model_training_evaluation.py # Model training and selection
├── 4_streamlit_app.py          # Web application interface
├── run_complete_pipeline.py    # Complete pipeline runner
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── adult_data.csv             # Your dataset (place here)
```

## 🔄 Usage

### Option 1: Run Complete Pipeline
```bash
python run_complete_pipeline.py
```

### Option 2: Run Individual Modules
```bash
# Step 1: Data Setup
python 1_setup_data.py

# Step 2: Data Preprocessing
python 2_data_preprocessing.py

# Step 3: Model Training
python 3_model_training_evaluation.py

# Step 4: Launch Web App
streamlit run 4_streamlit_app.py
```

### For Google Colab

1. Upload all Python files to your Colab environment
2. Upload your CSV dataset as `adult_data.csv`
3. Install requirements:
   ```python
   !pip install -r requirements.txt
   ```
4. Run the complete pipeline:
   ```python
   !python run_complete_pipeline.py
   ```
5. Launch Streamlit app:
   ```python
   !streamlit run 4_streamlit_app.py &
   ```

## 📊 Data Requirements

Your CSV dataset should contain the following columns:
- `age`: Age of the individual
- `workclass`: Type of employer
- `education`: Education level
- `marital-status`: Marital status
- `occupation`: Job occupation
- `relationship`: Relationship status
- `race`: Racial group
- `gender`: Gender
- `capital-gain`: Capital gains
- `capital-loss`: Capital losses
- `hours-per-week`: Hours worked per week
- `native-country`: Country of origin
- `income`: Target variable (<=50K or >50K)

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing
- **Missing Value Handling**: Intelligent imputation strategies
- **Feature Engineering**: Creation of derived features
- **Categorical Encoding**: One-hot and ordinal encoding
- **Feature Scaling**: Standardization of numerical features
- **Data Validation**: Comprehensive data quality checks

### 2. Model Training
- **Algorithm Comparison**: Tests multiple classification algorithms
- **Hyperparameter Tuning**: Optimizes best-performing models
- **Cross-Validation**: Robust model evaluation
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC AUC

### 3. Model Selection
- **Automated Selection**: Chooses best model based on F1-Score
- **Feature Importance**: Identifies key predictive factors
- **Model Persistence**: Saves trained models and preprocessors

## 🎨 Web Application Features

- **Interactive Input Form**: User-friendly interface for data entry
- **Real-time Predictions**: Instant salary bracket predictions
- **Confidence Scores**: Probability estimates for predictions
- **Feature Importance Visualization**: Charts showing key factors
- **Model Performance Metrics**: Displays model accuracy and statistics
- **Responsive Design**: Works on desktop and mobile devices

## 📈 Model Performance

The system automatically evaluates multiple algorithms:
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- XGBoost (if available)
- LightGBM (if available)
- Support Vector Machines
- K-Nearest Neighbors

## 🔧 Customization

### Adding New Features
1. Modify the feature engineering section in `2_data_preprocessing.py`
2. Update the input form in `4_streamlit_app.py`
3. Retrain the model using `3_model_training_evaluation.py`

### Changing Algorithms
1. Add new algorithms to the models dictionary in `3_model_training_evaluation.py`
2. Define hyperparameter grids for tuning
3. Update the model comparison logic

## 🚨 Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **File Not Found Errors**
   - Ensure your CSV file is named `adult_data.csv`
   - Check that all Python files are in the same directory

3. **Memory Issues**
   - Reduce dataset size for testing
   - Use fewer algorithms in model comparison

4. **Streamlit Issues**
   ```bash
   streamlit run 4_streamlit_app.py --server.port 8501
   ```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For questions or issues, please create an issue in the repository or contact the development team.

---

**Happy Predicting! 🎯**
