# tuning_utils.py

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def hyperparameter_tuning(X_train, y_train):
    """Performs hyperparameter tuning for Random Forest, Gradient Boosting, and SVR"""
    print("\n===== HYPERPARAMETER TUNING =====\n")

    # Random Forest hyperparameter tuning
    print("Tuning Random Forest...")
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestRegressor(random_state=42)
    gs_rf = GridSearchCV(rf, rf_param_grid, cv=5, scoring='r2', n_jobs=-1)
    gs_rf.fit(X_train, y_train)
    print(f"Best RF parameters: {gs_rf.best_params_}")
    print(f"Best RF R2 score: {gs_rf.best_score_:.4f}")

    # Gradient Boosting hyperparameter tuning
    print("\nTuning Gradient Boosting...")
    gbr_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gbr = GradientBoostingRegressor(random_state=42)
    gs_gbr = GridSearchCV(gbr, gbr_param_grid, cv=5, scoring='r2', n_jobs=-1)
    gs_gbr.fit(X_train, y_train)
    print(f"Best GBR parameters: {gs_gbr.best_params_}")
    print(f"Best GBR R2 score: {gs_gbr.best_score_:.4f}")

    # SVR hyperparameter tuning
    print("\nTuning SVR...")
    svr_param_grid = {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2],
        'kernel': ['rbf', 'linear']
    }
    svr_pipeline = make_pipeline(StandardScaler(), SVR())
    svr_param_grid_pipeline = {f'svr__{key}': value for key, value in svr_param_grid.items()}
    gs_svr = GridSearchCV(svr_pipeline, svr_param_grid_pipeline, cv=5, scoring='r2', n_jobs=-1)
    gs_svr.fit(X_train, y_train)
    print(f"Best SVR parameters: {gs_svr.best_params_}")
    print(f"Best SVR R2 score: {gs_svr.best_score_:.4f}")

    return gs_rf.best_estimator_, gs_gbr.best_estimator_, gs_svr.best_estimator_
