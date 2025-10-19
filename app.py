import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE

# --- Page customization ---
st.set_page_config(page_title="Prostate Cancer ML Dashboard", page_icon="🧬", layout="centered")
st.title("🧬 Prostate Cancer ML Dashboard")
st.markdown("""
Interactively compare machine learning models for prostate cancer.
Use the sidebar to choose a model and adjust test size or random state.
""")

@st.cache_data
def load_data():
    df = pd.read_excel('merged_prostrate_cancer_data.xlsx').dropna(subset=['categories']).drop(columns=['findings'])
    X = pd.get_dummies(df.drop(columns=['categories']), drop_first=True)
    y = df['categories'].map({'Benign': 1, 'Malignant': 0})
    return X, y

X, y = load_data()
st.write("#### Data Sample", X.head())

model_choice = st.sidebar.selectbox(
    "Choose model",
    ['Random Forest', 'SVC', 'Gradient Boosting', 'Voting Ensemble']
)
test_size = st.sidebar.slider("Test Size (fraction)", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", min_value=1, max_value=1000, value=42)

sm = SMOTE(random_state=random_state)
X_res, y_res = sm.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=test_size, random_state=random_state, stratify=y_res
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def plot_roc_curve_multi(curves):
    plt.style.use('default')  # Safe built-in style
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a']
    for idx, (label, y_test, y_prob) in enumerate(curves):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=colors[idx % len(colors)], label=f'{label} (AUC={roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves', fontsize=15, color='#37474f')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    fig.patch.set_facecolor('#eff6fa')
    return fig

def plot_roc_curve(y_test, y_prob, label, color='#2e7d32'):
    plt.style.use('default')
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, lw=2, color=color, label=f'{label} (AUC={roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, color='#37474f')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.patch.set_facecolor('#eff6fa')
    return fig, roc_auc

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
    auc_svc = roc_auc_score(y_test, y_prob_svc)
    auc_rf = roc_auc_score(y_test, y_prob_rf)
    auc_gbc = roc_auc_score(y_test, y_prob_gbc)
    auc_ens = roc_auc_score(y_test, y_prob_ens)
    cm = confusion_matrix(y_test, ensemble.predict(X_test_scaled))
    comparison_df = pd.DataFrame({
        'Actual': np.array(y_test)[:5],
        'Predicted': ensemble.predict(X_test_scaled)[:5]
    })
    fpr_svc, tpr_svc, _ = roc_curve(y_test, y_prob_svc)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    fpr_gbc, tpr_gbc, _ = roc_curve(y_test, y_prob_gbc)
    fpr_ens, tpr_ens, _ = roc_curve(y_test, y_prob_ens)
    return {
        "classification_reports": {
            "SVC": classification_report(y_test, svc_model.predict(X_test_scaled)),
            "RF": classification_report(y_test, rf_model.predict(X_test_scaled)),
            "GBC": classification_report(y_test, gbc_model.predict(X_test_scaled)),
            "Ensemble": classification_report(y_test, ensemble.predict(X_test_scaled))
        },
        "auc_scores": {
            "SVC": auc_svc, "RF": auc_rf, "GBC": auc_gbc, "Ensemble": auc_ens
        },
        "confusion_matrix": cm,
        "comparison_df": comparison_df,
        "roc_curves": [
            ("SVC", fpr_svc, tpr_svc, auc_svc, '--', '#1b9e77'),
            ("RF", fpr_rf, tpr_rf, auc_rf, '-.', '#d95f02'),
            ("GBC", fpr_gbc, tpr_gbc, auc_gbc, ':', '#7570b3'),
            ("Ensemble", fpr_ens, tpr_ens, auc_ens, '-', 'green')
        ]
    }

if model_choice == 'Random Forest':
    st.subheader("🌲 Random Forest Classifier (with Randomized Search)")
    rf = RandomForestClassifier(random_state=random_state)
    param_dist = {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
    rf_search = RandomizedSearchCV(
        rf, param_dist, n_iter=4, cv=3, scoring='roc_auc', random_state=random_state, n_jobs=-1
    )
    rf_search.fit(X_train, y_train)
    rf = rf_search.best_estimator_
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    st.write("##### Classification Report:")
    st.code(classification_report(y_test, y_pred))
    st.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
    st.write("##### Confusion Matrix:")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))
    fig, roc_auc = plot_roc_curve(y_test, y_prob, "Random Forest", color='#1b9e77')
    st.pyplot(fig)

elif model_choice == 'SVC':
    st.subheader("🔷 Support Vector Classifier")
    svc = SVC(probability=True, random_state=random_state)
    svc.fit(X_train_scaled, y_train)
    y_pred = svc.predict(X_test_scaled)
    y_prob = svc.predict_proba(X_test_scaled)[:, 1]
    st.write("##### Classification Report:")
    st.code(classification_report(y_test, y_pred))
    st.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
    st.write("##### Confusion Matrix:")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))
    fig, roc_auc = plot_roc_curve(y_test, y_prob, "SVC", color='#d95f02')
    st.pyplot(fig)

elif model_choice == 'Gradient Boosting':
    st.subheader("🌟 Gradient Boosting Classifier")
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=random_state)
    gbc.fit(X_train, y_train)
    y_pred = gbc.predict(X_test)
    y_prob = gbc.predict_proba(X_test)[:, 1]
    st.write("##### Classification Report:")
    st.code(classification_report(y_test, y_pred))
    st.metric("ROC AUC", f"{roc_auc_score(y_test, y_prob):.4f}")
    st.write("##### Confusion Matrix:")
    st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred)))
    fig, roc_auc = plot_roc_curve(y_test, y_prob, "GBC", color='#7570b3')
    st.pyplot(fig)

elif model_choice == 'Voting Ensemble':
    st.subheader("🤝 VotingClassifier Soft Ensemble")
    results = evaluate_voting_ensemble(X_train_scaled, X_test_scaled, y_train, y_test)
    st.metric("Ensemble ROC AUC", f"{results['auc_scores']['Ensemble']:.4f}")
    st.write("##### Confusion Matrix:")
    st.dataframe(pd.DataFrame(results['confusion_matrix']))
    st.write("##### First 5 Actual vs Predicted:")
    st.dataframe(results['comparison_df'])
    for model in ['SVC', 'RF', 'GBC', 'Ensemble']:
        st.write(f"###### {model} Classification Report:")
        st.code(results['classification_reports'][model])
    fig, ax = plt.subplots(figsize=(8,6))
    for label, fpr, tpr, auc_val, style, color in results['roc_curves']:
        ax.plot(fpr, tpr, linestyle=style, color=color, lw=2+(label=='Ensemble'),
                label=f"{label} (AUC={auc_val:.2f})")
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve Comparison", fontsize=15, color='#37474f')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.patch.set_facecolor("#eff6fa")
    st.pyplot(fig)
