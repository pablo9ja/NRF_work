import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    StackingRegressor, VotingRegressor,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report
)
from tuning_utils import hyperparameter_tuning

def preprocess_data(df, target_col='categories', binary_cols=None):
    features = [
        'age', 'state', 'marital_status', 'highest_level_of_education',
        'occupation', 'monthly_income', 'ethnic_group', 'who_made_you_seek_medical_help',
        'height_cm', 'weight', 'bmi', 'engagement_in_physical_exercise',
        'overall_diet', 'meals_taken_in_a_day', 'do_you_consume_alcohol',
        'do_you_smoke', 'how_do_you_rate_your_stress_level',
        'do_you_engage_in_activities_to_manage_stress',
        'hours_of_sleep_per_night',
        'types_of_cancer', 'types_of_biopsy', 'side',
        'symptions'
    ]
    if 'bmi' not in df.columns and all(col in df.columns for col in ['weight', 'height_cm']):
        df['bmi'] = df['weight'] / (df['height_cm'] / 100) ** 2
    missing = [col for col in features if col not in df.columns]
    if missing:
        print("The following features are missing in your data and will be skipped:", missing)
    features = [col for col in features if col in df.columns]
    if binary_cols is None:
        binary_cols = ['do_you_consume_alcohol', 'do_you_smoke', 'do_you_engage_in_activities_to_manage_stress']
    for col in binary_cols:
        if col in df.columns:
            df[col] = LabelEncoder().fit_transform(df[col])
    categorical_cols = [
        col for col in df.select_dtypes(include=['object', 'category']).columns
        if col not in binary_cols and col != target_col
    ]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    if target_col in df.columns:
        df[target_col] = LabelEncoder().fit_transform(df[target_col])
    else:
        raise ValueError(f"Target column '{target_col}' not found in the dataframe.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y, df

def train_tuned_models(X_train, X_test, y_train, y_test):
    best_rf, best_gbr, best_svr = hyperparameter_tuning(X_train, y_train)
    results = {}
    # Train tuned Random Forest
    print("\nTraining tuned Random Forest...")
    best_rf.fit(X_train, y_train)
    rf_pred = best_rf.predict(X_test)
    results['RandomForest_Tuned'] = {
        'model': best_rf, 'predictions': rf_pred,
        'MAE': mean_absolute_error(y_test, rf_pred),
        'MSE': mean_squared_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R²': r2_score(y_test, rf_pred)
    }
    # Train tuned Gradient Boosting
    print("Training tuned Gradient Boosting...")
    best_gbr.fit(X_train, y_train)
    gbr_pred = best_gbr.predict(X_test)
    results['GBR_Tuned'] = {
        'model': best_gbr, 'predictions': gbr_pred,
        'MAE': mean_absolute_error(y_test, gbr_pred),
        'MSE': mean_squared_error(y_test, gbr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, gbr_pred)),
        'R²': r2_score(y_test, gbr_pred)
    }
    # Train tuned SVR
    print("Training tuned SVR...")
    best_svr.fit(X_train, y_train)
    svr_pred = best_svr.predict(X_test)
    results['SVR_Tuned'] = {
        'model': best_svr, 'predictions': svr_pred,
        'MAE': mean_absolute_error(y_test, svr_pred),
        'MSE': mean_squared_error(y_test, svr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, svr_pred)),
        'R²': r2_score(y_test, svr_pred)
    }
    # Train ensemble with tuned models
    print("Training ensemble with tuned models...")
    ensemble_tuned = VotingRegressor(estimators=[
        ('rf', best_rf), ('gbr', best_gbr), ('svr', best_svr)
    ])
    ensemble_tuned.fit(X_train, y_train)
    ensemble_pred = ensemble_tuned.predict(X_test)
    results['Ensemble_Tuned'] = {
        'model': ensemble_tuned, 'predictions': ensemble_pred,
        'MAE': mean_absolute_error(y_test, ensemble_pred),
        'MSE': mean_squared_error(y_test, ensemble_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
        'R²': r2_score(y_test, ensemble_pred)
    }
    # Train stacking with tuned models
    print("Training stacking with tuned models...")
    stacking_tuned = StackingRegressor(
        estimators=[
            ('rf', best_rf), ('gbr', best_gbr), ('svr', best_svr)
        ],
        final_estimator=RidgeCV(),
        passthrough=True
    )
    stacking_tuned.fit(X_train, y_train)
    stacking_pred = stacking_tuned.predict(X_test)
    results['StackingRegressor_Tuned'] = {
        'model': stacking_tuned, 'predictions': stacking_pred,
        'MAE': mean_absolute_error(y_test, stacking_pred),
        'MSE': mean_squared_error(y_test, stacking_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, stacking_pred)),
        'R²': r2_score(y_test, stacking_pred)
    }
    return results

def train_all_models(df, target_col='categories', binary_cols=None, test_size=0.2, random_state=42):
    X, y, processed_df = preprocess_data(df, target_col, binary_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("\nTraining tuned models...")
    results = train_tuned_models(X_train, X_test, y_train, y_test)
    return results, y_test

def print_metrics_and_classification(results, y_test):
    print("\n===== MODEL EVALUATION AND CLASSIFICATION METRICS =====\n")
    for model_name, metrics in results.items():
        y_pred = metrics['predictions']
        print(f"\n{model_name} Regression Metrics:")
        print(f"  - Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
        print(f"  - Mean Squared Error (MSE): {metrics['MSE']:.4f}")
        print(f"  - Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
        print(f"  - R² Score: {metrics['R²']:.4f}")
        print("  (First 10 Predicted vs True):")
        comp_df = pd.DataFrame({
            'True Values': y_test.values,
            'Predicted Values': y_pred
        })
        print(comp_df.head(10))
        # Classification thresholding
        y_pred_binary = (y_pred > 0.5).astype(int)
        print(f"\n{model_name} Classification Metrics (Threshold=0.5):")
        print(f"  - Accuracy: {accuracy_score(y_test, y_pred_binary):.4f}")
        print(f"  - Confusion Matrix:\n{confusion_matrix(y_test, y_pred_binary)}")
        print(f"  - Classification Report:\n{classification_report(y_test, y_pred_binary)}")
        # Scatter plot
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test, y_pred, alpha=0.7, edgecolor='k')
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal')
        plt.title(f"{model_name} - Predicted vs Actual")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.tight_layout()
        plt.show()

def print_performance_table_and_plot(results):
    rows = []
    for model_name, metrics in results.items():
        if 'MAE' in metrics:  # regression model
            rows.append({
                'Model': model_name,
                'MAE': round(metrics['MAE'], 4),
                'MSE': round(metrics['MSE'], 4),
                'RMSE': round(metrics['RMSE'], 4),
                'R²': round(metrics['R²'], 4)
            })
    df_results = pd.DataFrame(rows)
    print("\nPerformance Summary Table:\n")
    print(df_results.to_string(index=False))
    metrics_names = ['MAE', 'MSE', 'RMSE', 'R²']
    x = np.arange(len(df_results['Model']))
    bar_width = 0.2
    plt.figure(figsize=(10,6))
    for i, metric in enumerate(metrics_names):
        plt.bar(x + i*bar_width, df_results[metric], width=bar_width, label=metric)
    plt.xticks(x + bar_width*(len(metrics_names)-1)/2, df_results['Model'])
    plt.ylabel("Metric Value")
    plt.title("Regression Model Performance Comparison")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    df = pd.read_excel('merged_breast_cancer_data.xlsx')
    target = 'categories'
    results, y_test = train_all_models(df, target_col=target, test_size=0.2, random_state=42)
    print_metrics_and_classification(results, y_test)
    print_performance_table_and_plot(results)
