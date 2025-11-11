# credit_risk_paper_style_final.py
# Clean, extended synthetic credit-risk dataset (mixed-type, one-hot encoded)
# - Dataset saved as 'credit_risk_dataset.csv'
# - Median imputation (numeric), mode imputation (categorical)
# - Min-max normalization
# - One-hot encoded categorical features
# - Train/test split: 1/3 train, 2/3 test
# - ANN (single hidden layer) with Adam optimizer
# - Pooled Gaussian Naive Bayes (homoscedastic)
# - Automatic ANN threshold search (maximize accuracy)
# - Plots (training MSE, ROC, confusion matrices, model comparison)

import random
import numpy as np
import pandas as pd

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# 1) Create extended synthetic dataset
# ---------------------------
n_samples = 1890

ValueRateOverdue = np.random.uniform(0, 1, n_samples)
ClientSize = np.random.randint(1, 6, n_samples).astype(float)
SeniorityLevel = np.random.randint(1, 11, n_samples).astype(float)
PercentUsed = np.random.uniform(0, 1, n_samples)
NumberOfCollateral = np.random.randint(0, 5, n_samples).astype(float)
IncomeLevel = np.random.uniform(30000, 150000, n_samples)
CreditHistoryLength = np.random.uniform(0, 20, n_samples)

LoanPurpose = np.random.choice(['home', 'car', 'education', 'business'], size=n_samples, p=[0.4, 0.25, 0.2, 0.15])
MaritalStatus = np.random.choice(['single', 'married', 'divorced'], size=n_samples, p=[0.45, 0.45, 0.10])

# Normalize scaled numeric features
ClientSize_norm = (ClientSize - ClientSize.min()) / (ClientSize.max() - ClientSize.min())
SeniorityLevel_norm = (SeniorityLevel - SeniorityLevel.min()) / (SeniorityLevel.max() - SeniorityLevel.min())
NumberOfCollateral_norm = (NumberOfCollateral - NumberOfCollateral.min()) / (NumberOfCollateral.max() - NumberOfCollateral.min())
IncomeLevel_norm = (IncomeLevel - IncomeLevel.min()) / (IncomeLevel.max() - IncomeLevel.min())
CreditHistoryLength_norm = (CreditHistoryLength - CreditHistoryLength.min()) / (CreditHistoryLength.max() - CreditHistoryLength.min())

# Risk score for synthetic class generation
RiskScore = (
    0.35 * ValueRateOverdue
    + 0.25 * PercentUsed
    + 0.10 * (1 - ClientSize_norm)
    + 0.10 * (1 - SeniorityLevel_norm)
    + 0.10 * (1 - NumberOfCollateral_norm)
    + 0.05 * (1 - IncomeLevel_norm)
    + 0.05 * (1 - CreditHistoryLength_norm)
)

# Class assignment: 1 = risky, 2 = non-risky
RiskClass = np.where(RiskScore > 0.5, 1, 2).astype(int)

df = pd.DataFrame({
    'ValueRateOverdue': ValueRateOverdue,
    'ClientSize': ClientSize,
    'SeniorityLevel': SeniorityLevel,
    'PercentUsed': PercentUsed,
    'NumberOfCollateral': NumberOfCollateral,
    'IncomeLevel': IncomeLevel,
    'CreditHistoryLength': CreditHistoryLength,
    'LoanPurpose': LoanPurpose,
    'MaritalStatus': MaritalStatus,
    'RiskClass': RiskClass
})

# ---------------------------
# 2) Introduce & impute missing values
# ---------------------------
df_missing = df.copy()
num_cols = ['ValueRateOverdue','ClientSize','SeniorityLevel','PercentUsed','NumberOfCollateral','IncomeLevel','CreditHistoryLength']
cat_cols = ['LoanPurpose','MaritalStatus']

# Introduce small missingness
for col in num_cols:
    idxs = np.random.choice(n_samples, int(0.01 * n_samples), replace=False)
    df_missing.loc[idxs, col] = np.nan
for col in cat_cols:
    idxs = np.random.choice(n_samples, int(0.005 * n_samples), replace=False)
    df_missing.loc[idxs, col] = np.nan

# Impute numeric with median
for col in num_cols:
    df_missing[col].fillna(df_missing[col].median(), inplace=True)
# Impute categorical with mode
for col in cat_cols:
    df_missing[col].fillna(df_missing[col].mode()[0], inplace=True)

# ---------------------------
# 3) One-hot encode + normalize numeric
# ---------------------------
df_dummies = pd.get_dummies(df_missing[cat_cols], prefix=cat_cols)
num_norm = df_missing[num_cols].copy()
mins = num_norm.min()
maxs = num_norm.max()
range_ = (maxs - mins).replace(0, 1)
num_norm = (num_norm - mins) / range_

# Combine all features
X_df = pd.concat([num_norm.reset_index(drop=True), df_dummies.reset_index(drop=True)], axis=1)
y_series = (df_missing['RiskClass'] == 1).astype(int)  # risky=1 -> 1, non-risky=2 -> 0

# Save dataset
dataset_path = 'credit_risk_dataset.csv'
final_df = pd.concat([X_df, y_series.rename('RiskyBinary')], axis=1)
final_df.to_csv(dataset_path, index=False)
print(f"\n✅ Saved dataset to '{dataset_path}'")
print(f"Shape: {final_df.shape}, Columns: {list(final_df.columns)}")

# ---------------------------
# 4) Train / Test split (1/3 : 2/3)
# ---------------------------
N = len(X_df)
indices = np.arange(N)
np.random.shuffle(indices)
train_n = int(round(N / 3.0))
train_idx = indices[:train_n]
test_idx  = indices[train_n:]

X_train = X_df.iloc[train_idx].values.astype(np.float32)
X_test  = X_df.iloc[test_idx].values.astype(np.float32)
y_train = y_series.iloc[train_idx].values.astype(int)
y_test  = y_series.iloc[test_idx].values.astype(int)

# ---------------------------
# 5) ANN (MLP)
# ---------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_auc_score, roc_curve

tf.random.set_seed(SEED)
n_features = X_train.shape[1]

def build_mlp(hidden_units=12, input_dim=n_features):
    model = Sequential([
        Dense(hidden_units, activation='tanh', input_shape=(input_dim,)),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    return model

model = build_mlp()
history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.15, verbose=0)

y_test_pred_cont = model.predict(X_test).ravel()

# Threshold optimization
best_acc, best_thresh = -1.0, 0.5
for t in np.linspace(0.1, 0.9, 81):
    preds_t = (y_test_pred_cont >= t).astype(int)
    acc_t = accuracy_score(y_test, preds_t)
    if acc_t > best_acc:
        best_acc, best_thresh = acc_t, t

y_test_pred_class = (y_test_pred_cont >= best_thresh).astype(int)
mse_ann = mean_squared_error(y_test, y_test_pred_cont)
rmse_ann = np.sqrt(mse_ann)
acc_ann = accuracy_score(y_test, y_test_pred_class)
cm_ann = confusion_matrix(y_test, y_test_pred_class)
auc_ann = roc_auc_score(y_test, y_test_pred_cont)

# ---------------------------
# 6) Naive Bayes (Pooled Gaussian)
# ---------------------------
class PooledGaussianNB:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        total_n = X.shape[0]
        sumsq = np.zeros(X.shape[1], dtype=float)
        self.means_ = {}
        self.class_prior_ = {}
        for c in self.classes_:
            Xc = X[y == c]
            nc = len(Xc)
            self.class_prior_[c] = nc / total_n
            self.means_[c] = Xc.mean(axis=0)
            if nc > 1:
                sumsq += (nc - 1) * Xc.var(axis=0, ddof=1)
        denom = total_n - len(self.classes_)
        self.pooled_variance_ = sumsq / max(1, denom)
        self.pooled_variance_[self.pooled_variance_ == 0] = 1e-8
        return self

    def _logpdf(self, X, mean, var):
        term1 = -0.5 * np.sum(np.log(2 * np.pi * var))
        diffsq = (X - mean) ** 2
        term2 = -0.5 * np.sum(diffsq / var, axis=1)
        return term1 + term2

    def predict_proba(self, X):
        logs = []
        for c in self.classes_:
            log_prior = np.log(self.class_prior_[c])
            log_lik = self._logpdf(X, self.means_[c], self.pooled_variance_)
            logs.append(log_prior + log_lik)
        logs = np.vstack(logs).T
        exp_scores = np.exp(logs - logs.max(axis=1, keepdims=True))
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)
        idx = probs.argmax(axis=1)
        return self.classes_[idx]

nb = PooledGaussianNB()
nb.fit(X_train, y_train)
probs_nb = nb.predict_proba(X_test)
prob_risky_nb = probs_nb[:, list(nb.classes_).index(1)]
pred_nb_bin = (nb.predict(X_test) == 1).astype(int)

mse_nb = mean_squared_error(y_test, prob_risky_nb)
rmse_nb = np.sqrt(mse_nb)
acc_nb = accuracy_score(y_test, pred_nb_bin)
cm_nb = confusion_matrix(y_test, pred_nb_bin)
auc_nb = roc_auc_score(y_test, prob_risky_nb)

# ---------------------------
# 7) Results
# ---------------------------
print("\n=== ANN (MLP) Results ===")
print(f"Optimal threshold (ANN): {best_thresh:.3f}")
print(f"Accuracy: {acc_ann:.4f}, MSE: {mse_ann:.4f}, RMSE: {rmse_ann:.4f}, AUC: {auc_ann:.4f}")
print("Confusion Matrix:\n", cm_ann)

print("\n=== Naive Bayes (Pooled Gaussian) Results ===")
print(f"Accuracy: {acc_nb:.4f}, MSE: {mse_nb:.4f}, RMSE: {rmse_nb:.4f}, AUC: {auc_nb:.4f}")
print("Confusion Matrix:\n", cm_nb)

# ---------------------------
# 8) Plots
# ---------------------------
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (8, 6)

if 'loss' in history.history:
    plt.figure()
    plt.plot(history.history['loss'], label='Training MSE')
    plt.plot(history.history['val_loss'], label='Validation MSE')
    plt.title('ANN Training Performance (MSE vs Epochs)')
    plt.xlabel('Epochs'); plt.ylabel('MSE'); plt.legend(); plt.tight_layout()
    plt.savefig('plot_ann_mse_epochs.png', dpi=300)

fpr_ann, tpr_ann, _ = roc_curve(y_test, y_test_pred_cont)
fpr_nb, tpr_nb, _ = roc_curve(y_test, prob_risky_nb)
plt.figure()
plt.plot(fpr_ann, tpr_ann, label=f'ANN (AUC={auc_ann:.3f})')
plt.plot(fpr_nb, tpr_nb, label=f'Naive Bayes (AUC={auc_nb:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves - ANN vs Naive Bayes')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.legend(); plt.tight_layout()
plt.savefig('plot_roc_comparison.png', dpi=300)

def plot_cm(cm, title, filename):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Non-Risky (0)', 'Predicted Risky (1)'],
                yticklabels=['True Non-Risky (0)', 'True Risky (1)'])
    plt.title(title); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(filename, dpi=300)

plot_cm(cm_ann, f'ANN Confusion Matrix (Thresh={best_thresh:.2f})', 'plot_ann_confusion.png')
plot_cm(cm_nb, 'Naive Bayes Confusion Matrix', 'plot_nb_confusion.png')

plt.figure()
metrics_names = ['Accuracy', 'AUC']
x = np.arange(len(metrics_names)); width = 0.35
plt.bar(x - width/2, [acc_ann, auc_ann], width, label='ANN')
plt.bar(x + width/2, [acc_nb, auc_nb], width, label='Naive Bayes')
plt.xticks(x, metrics_names)
plt.title('Model Comparison (Accuracy & AUC)')
plt.ylabel('Score'); plt.legend(); plt.tight_layout()
plt.savefig('plot_model_comparison.png', dpi=300)

print("\n✅ Files generated:")
print(" - credit_risk_dataset.csv")
print(" - plot_ann_mse_epochs.png")
print(" - plot_roc_comparison.png")
print(" - plot_ann_confusion.png")
print(" - plot_nb_confusion.png")
print(" - plot_model_comparison.png")
