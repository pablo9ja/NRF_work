import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, auc,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
)

# ---- Data Preparation ----
def prepare_data(filepath):
    df = pd.read_excel(filepath)
    df = df.dropna(subset=['categories'])
    df = df.drop(columns=['findings'])
    X = df.drop(columns=['categories'])
    y_encoded = df['categories'].map({'Benign': 1, 'Malignant': 0})
    X_encoded = pd.get_dummies(X, drop_first=True)
    return X_encoded, y_encoded

# ---- RandomForest Evaluation ----
def rf_model_evaluation(X_encoded, y_encoded):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_encoded, y_encoded)
    X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )
    param_dist = {
        'n_estimators': [100, 200, 500, 800],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'bootstrap': [True, False]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf_search = RandomizedSearchCV(
        rf, param_distributions=param_dist,
        n_iter=50, cv=cv, scoring='roc_auc',
        random_state=42, n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train_res, y_train_res)
    best_rf = rf_search.best_estimator_
    print("Best Parameters:", rf_search.best_params_)
    y_pred_rf = best_rf.predict(X_test_res)
    y_prob_rf = best_rf.predict_proba(X_test_res)[:, 1]
    print("\nRandom Forest Classification Report:\n", classification_report(y_test_res, y_pred_rf))
    print("Random Forest Test ROC AUC:", roc_auc_score(y_test_res, y_prob_rf))
    cm = confusion_matrix(y_test_res, y_pred_rf)
    print("\nConfusion Matrix:\n", cm)
    importances = best_rf.feature_importances_
    feat_names = X_encoded.columns if hasattr(X_encoded, 'columns') else [f"Feature {i}" for i in range(X_encoded.shape[1])]
    print("\nTop Feature Importances:")
    for name, val in sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{name}: {val:.3f}")
    fpr, tpr, thresholds = roc_curve(y_test_res, y_prob_rf)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkblue', lw=2, label=f'RF ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Random Forest ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return best_rf, rf_search, X_train_res, X_test_res, y_train_res, y_test_res

# ---- SVC Evaluation ----
def train_evaluate_svc(X_encoded, y_encoded, test_size=0.2, random_state=42):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_encoded, y_encoded)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=test_size, random_state=random_state, stratify=y_res
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svc_model = SVC(probability=True, random_state=random_state)
    svc_model.fit(X_train_scaled, y_train)
    y_pred = svc_model.predict(X_test_scaled)
    y_prob = svc_model.predict_proba(X_test_scaled)[:, 1]
    print("\nSVC Classification Report:\n", classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"SVC ROC AUC: {roc_auc:.4f}")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'SVC ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.title('ROC Curve - SVC', fontsize=13)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return svc_model, scaler, X_train_scaled, X_test_scaled, y_train, y_test

# ---- Gradient Boosting Evaluation ----
def train_evaluate_gbc(X_encoded, y_encoded, test_size=0.2, random_state=42):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_encoded, y_encoded)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=test_size, random_state=random_state, stratify=y_res
    )
    gbc_model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, random_state=random_state
    )
    gbc_model.fit(X_train, y_train)
    y_pred = gbc_model.predict(X_test)
    y_prob = gbc_model.predict_proba(X_test)[:, 1]
    print("\nGradient Boosting Classification Report:\n", classification_report(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"GBC ROC AUC: {roc_auc:.4f}")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'GBC ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('ROC Curve - Gradient Boosting', fontsize=13)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    return gbc_model, X_train, X_test, y_train, y_test

# ---- Voting Ensemble Evaluation ----
def evaluate_voting_ensemble(X_train_scaled, X_test_scaled, y_train, y_test, best_rf=None):
    svc_model = SVC(probability=True, random_state=42)
    rf_model = best_rf if best_rf is not None else RandomForestClassifier(n_estimators=100, random_state=42)
    gbc_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    svc_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train_scaled, y_train)
    gbc_model.fit(X_train_scaled, y_train)
    ensemble = VotingClassifier(
        estimators=[('svc', svc_model), ('rf', rf_model), ('gbc', gbc_model)],
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train)
    y_prob_svc = svc_model.predict_proba(X_test_scaled)[:, 1]
    y_prob_rf  = rf_model.predict_proba(X_test_scaled)[:, 1]
    y_prob_gbc = gbc_model.predict_proba(X_test_scaled)[:, 1]
    y_prob_ens = ensemble.predict_proba(X_test_scaled)[:, 1]
    print("\nSVC Report:\n", classification_report(y_test, svc_model.predict(X_test_scaled)))
    print("\nRandom Forest Report:\n", classification_report(y_test, rf_model.predict(X_test_scaled)))
    print("\nGradient Boost Report:\n", classification_report(y_test, gbc_model.predict(X_test_scaled)))
    print("\nEnsemble Report:\n", classification_report(y_test, ensemble.predict(X_test_scaled)))
    auc_svc = roc_auc_score(y_test, y_prob_svc)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    auc_gbc = roc_auc_score(y_test, y_prob_gbc)
    auc_ens = roc_auc_score(y_test, y_prob_ens)
    print(f"SVC ROC AUC: {auc_svc:.4f}")
    print(f"RF ROC AUC: {auc_rf:.4f}")
    print(f"GBC ROC AUC: {auc_gbc:.4f}")
    print(f"Ensemble ROC AUC: {auc_ens:.4f}")
    cm = confusion_matrix(y_test, ensemble.predict(X_test_scaled))
    print("\nEnsemble Confusion Matrix:\n", cm)
    fpr_svc, tpr_svc, _ = roc_curve(y_test, y_prob_svc)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    fpr_gbc, tpr_gbc, _ = roc_curve(y_test, y_prob_gbc)
    fpr_ens, tpr_ens, _ = roc_curve(y_test, y_prob_ens)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_svc, tpr_svc, lw=2, label=f'SVC (AUC = {auc_svc:.2f})', linestyle='--')
    plt.plot(fpr_rf, tpr_rf, lw=2, label=f'Random Forest (AUC = {auc_rf:.2f})', linestyle='-.')
    plt.plot(fpr_gbc, tpr_gbc, lw=2, label=f'Gradient Boosting (AUC = {auc_gbc:.2f})', linestyle=':')
    plt.plot(fpr_ens, tpr_ens, lw=3, color='green', label=f'Ensemble (AUC = {auc_ens:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison: Base Models vs Ensemble')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    comparison_df = pd.DataFrame({
        'Actual': np.array(y_test)[:5],
        'Predicted': ensemble.predict(X_test_scaled)[:5]
    })
    print("\nFirst 5 Actual vs Predicted:\n", comparison_df)
    return {
        "svc_model": svc_model,
        "rf_model": rf_model,
        "gbc_model": gbc_model,
        "ensemble": ensemble,
        "auc_scores": {
            "svc": auc_svc,
            "rf": auc_rf,
            "gbc": auc_gbc,
            "ensemble": auc_ens
        },
        "confusion_matrix": cm
    }

# ---- Main Experiment ----
if __name__ == "__main__":
    X_encoded, y_encoded = prepare_data('merged_prostrate_cancer_data.xlsx')
    print(X_encoded.head(2), '\n', y_encoded.head(2), '\n', y_encoded.isnull().sum())
    # RF
    best_rf, rf_search, X_train_rf, X_test_rf, y_train_rf, y_test_rf = rf_model_evaluation(X_encoded, y_encoded)
    # SVC
    svc_model, scaler, X_train_scaled, X_test_scaled, y_train_sm, y_test_sm = train_evaluate_svc(X_encoded, y_encoded)
    # GBC
    gbc_model, X_train_gbc, X_test_gbc, y_train_gbc, y_test_gbc = train_evaluate_gbc(X_encoded, y_encoded)
    ensemble_results = evaluate_voting_ensemble(
        X_train_scaled, X_test_scaled, y_train_sm, y_test_sm, best_rf=best_rf
    )
