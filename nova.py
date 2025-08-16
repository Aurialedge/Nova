import numpy as np
import pandas as pd
import os
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
pd.set_option('future.no_silent_downcasting', True)
np.random.seed(42)

def percentile_ranks(arr):
    arr = np.asarray(arr)
    if len(arr) <= 1:
        return np.zeros_like(arr, dtype=float)
    ranks = rankdata(arr, method='average') - 1
    return ranks / (len(arr) - 1)

def clamp(arr, lo=0.0, hi=100.0):
    return np.minimum(np.maximum(arr, lo), hi)


N_driver = 400
N_merchant = 350

drivers = pd.DataFrame({
    'role': 'driver',
    'on_time': np.random.normal(90, 5, N_driver).clip(60, 100),
    'earn_mean': np.random.normal(75, 12, N_driver).clip(30, 140),
    'earn_var': np.random.normal(10, 4, N_driver).clip(1, 40),
    'cancel_rate': np.random.uniform(0, 15, N_driver),
    'rating': np.random.normal(4.7, 0.2, N_driver).clip(3.5, 5.0),
    'hard_brake_rate': np.random.uniform(0, 8, N_driver),
    'trip_count_30d': np.random.normal(220, 60, N_driver).clip(10, 500),
    'days_on_platform': np.random.randint(30, 1200, N_driver),
    'pct_work_on_platform': np.random.uniform(0.4, 1.0, N_driver),
    'complaints_30d': np.random.poisson(0.2, N_driver)
})

merchants = pd.DataFrame({
    'role': 'merchant',
    'fulfill_rate': np.random.normal(87, 6, N_merchant).clip(50, 100),
    'revenue_mean': np.random.normal(65, 15, N_merchant).clip(10, 200),
    'revenue_var': np.random.normal(12, 5, N_merchant).clip(1, 60),
    'refund_rate': np.random.uniform(0, 12, N_merchant),
    'rating': np.random.normal(4.5, 0.25, N_merchant).clip(3.0, 5.0),
    'complaint_rate': np.random.uniform(0, 10, N_merchant),
    'order_count_30d': np.random.normal(300, 80, N_merchant).clip(5, 1500),
    'days_on_platform': np.random.randint(30, 1500, N_merchant),
    'pct_work_on_platform': np.random.uniform(0.2, 1.0, N_merchant),
    'complaints_30d': np.random.poisson(0.5, N_merchant)
})

df_raw = pd.concat([drivers, merchants], ignore_index=True)
df_raw.insert(0, 'partner_id', np.arange(1, len(df_raw)+1))
df_raw.head()

df = df_raw.copy()

mask_driver = df['role'] == 'driver'
mask_merchant = df['role'] == 'merchant'

df.loc[mask_driver, 'earn_cv'] = df.loc[mask_driver, 'earn_var'] / (df.loc[mask_driver, 'earn_mean'] + 1e-6)
df.loc[mask_driver, 'service_q'] = 0.6 * (df.loc[mask_driver, 'rating'] / 5.0) * 100 + 0.4 * df.loc[mask_driver, 'on_time'] - 0.7 * df.loc[mask_driver, 'cancel_rate']
df.loc[mask_driver, 'safe_driving'] = 100 - df.loc[mask_driver, 'hard_brake_rate'] * 6  

df.loc[mask_merchant, 'rev_cv'] = df.loc[mask_merchant, 'revenue_var'] / (df.loc[mask_merchant, 'revenue_mean'] + 1e-6)
df.loc[mask_merchant, 'service_q'] = 0.6 * (df.loc[mask_merchant, 'rating'] / 5.0) * 100 + 0.4 * df.loc[mask_merchant, 'fulfill_rate'] - 0.7 * df.loc[mask_merchant, 'refund_rate']
df.loc[mask_merchant, 'safe_driving'] = 100 - df.loc[mask_merchant, 'complaint_rate'] * 6  

df['activity_30d'] = np.where(mask_driver, df['trip_count_30d'], df['order_count_30d'])
df['rating_scaled'] = (df['rating'] / 5.0) * 100  

df.loc[mask_driver, 'S_B_raw'] = (
    0.4 * df.loc[mask_driver, 'on_time'] +
    0.25 * df.loc[mask_driver, 'safe_driving'] +
    0.2 * (100 - df.loc[mask_driver, 'cancel_rate']) + 
    0.15 * (100 - df.loc[mask_driver, 'complaints_30d']*10)
)

df.loc[mask_merchant, 'S_B_raw'] = (
    0.45 * df.loc[mask_merchant, 'fulfill_rate'] +
    0.25 * (100 - df.loc[mask_merchant, 'complaints_30d']*10) +
    0.30 * (100 - df.loc[mask_merchant, 'refund_rate'])
)

max_days = 1000.0
df['tenure_scaled'] = np.minimum(df['days_on_platform'] / max_days, 1.0) * 100
df['pct_work_scaled'] = df['pct_work_on_platform'] * 100
df['S_L_raw'] = 0.6 * df['tenure_scaled'] + 0.4 * df['pct_work_scaled']

for role in df['role'].unique():
    m = df['role'] == role
    cols_to_pct = ['on_time','earn_mean','earn_cv','service_q','rating_scaled','activity_30d','S_B_raw','S_L_raw','safe_driving','cancel_rate','fulfill_rate','revenue_mean','rev_cv','refund_rate','complaint_rate']
    for col in cols_to_pct:
        if col in df.columns:
            arr = df.loc[m, col].values
            if len(arr) > 0:
                df.loc[m, col + '_pctl'] = 100.0 * percentile_ranks(arr)
            else:
                df.loc[m, col + '_pctl'] = 0.0

df['S_B'] = clamp(df['S_B_raw_pctl'].fillna(50.0), 0, 100)
df['S_L'] = clamp(df['S_L_raw_pctl'].fillna(50.0), 0, 100)

df.loc[mask_driver, 'S_P'] = clamp(0.6 * df.loc[mask_driver, 'on_time_pctl'] + 0.4 * df.loc[mask_driver, 'service_q_pctl'], 0, 100)
df.loc[mask_merchant, 'S_P'] = clamp(0.6 * df.loc[mask_merchant, 'fulfill_rate_pctl'] + 0.4 * df.loc[mask_merchant, 'service_q_pctl'], 0, 100)

df.loc[mask_driver, 'S_I'] = clamp(0.6 * df.loc[mask_driver, 'earn_mean_pctl'] + 0.4 * (100 - df.loc[mask_driver, 'earn_cv_pctl']), 0, 100)
df.loc[mask_merchant, 'S_I'] = clamp(0.6 * df.loc[mask_merchant, 'revenue_mean_pctl'] + 0.4 * (100 - df.loc[mask_merchant, 'rev_cv_pctl']), 0, 100)

df['S_T'] = clamp(df['rating_scaled_pctl'].fillna(50.0), 0, 100)

df.loc[mask_driver, 'S_R'] = clamp(0.5 * (100 - df.loc[mask_driver, 'cancel_rate_pctl']) + 0.5 * df.loc[mask_driver, 'safe_driving_pctl'], 0, 100)
df.loc[mask_merchant, 'S_R'] = clamp(0.6 * (100 - df.loc[mask_merchant, 'complaint_rate_pctl']) + 0.4 * (100 - df.loc[mask_merchant, 'refund_rate_pctl']), 0, 100)

for col in ['S_P','S_I','S_T','S_R','S_B','S_L']:
    if col in df.columns:
        df[col] = df[col].fillna(50.0)

df[['partner_id','role','S_P','S_I','S_T','S_R','S_B','S_L']].head()

def make_latent_from_inputs(row, weights):
    vals = np.array([row[c] for c in weights.keys()], dtype=float)
    w = np.array(list(weights.values()), dtype=float)
    return float(np.clip((vals * w).sum() / w.sum() + np.random.normal(0, 4), 0, 100))

latent_P, latent_I, latent_T, latent_R, latent_B, latent_L = [], [], [], [], [], []
for _, r in df.iterrows():
    latent_P.append(make_latent_from_inputs(r, {'S_P':1.0, 'S_T':0.4}))
    latent_I.append(make_latent_from_inputs(r, {'S_I':1.0, 'S_P':0.2}))
    latent_T.append(make_latent_from_inputs(r, {'S_T':1.0}))
    latent_R.append(make_latent_from_inputs(r, {'S_R':1.0}))
    latent_B.append(make_latent_from_inputs(r, {'S_B':1.0}))
    latent_L.append(make_latent_from_inputs(r, {'S_L':1.0}))

df['latent_P'] = latent_P
df['latent_I'] = latent_I
df['latent_T'] = latent_T
df['latent_R'] = latent_R
df['latent_B'] = latent_B
df['latent_L'] = latent_L

bucket_models = {'driver': {}, 'merchant': {}}
buckets = ['P','I','T','R','B','L']
preds = {b: [] for b in buckets}

for role in ['driver','merchant']:
    m = df['role'] == role
    df_role = df.loc[m].reset_index(drop=True)
    for b in buckets:
        if b == 'P':
            in_cols = ['on_time_pctl','service_q_pctl','fulfill_rate_pctl','activity_30d_pctl']
        elif b == 'I':
            in_cols = ['earn_mean_pctl','revenue_mean_pctl','earn_cv_pctl','rev_cv_pctl']
        elif b == 'T':
            in_cols = ['rating_scaled_pctl']
        elif b == 'R':
            in_cols = ['safe_driving_pctl','cancel_rate_pctl','complaint_rate_pctl','refund_rate_pctl']
        elif b == 'B':
            in_cols = ['S_B_raw_pctl','service_q_pctl','complaints_30d']
        elif b == 'L':
            in_cols = ['S_L_raw_pctl','days_on_platform','pct_work_on_platform']
        else:
            in_cols = []
        in_cols = [c for c in in_cols if c in df_role.columns]
        X = df_role[in_cols].fillna(0).values
        y = df_role[f'latent_{b}'].values
        if len(X) < 10:
            class MeanModel:
                def __init__(self,val): self.val=val
                def predict(self,X): return np.full(len(X), self.val)
            mval = float(np.mean(y))
            model = MeanModel(mval)
        else:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
            gbr = GradientBoostingRegressor(random_state=42)
            rf  = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
            gbr.fit(X_tr, y_tr)
            rf.fit(X_tr, y_tr)
            r2_gbr = r2_score(y_te, gbr.predict(X_te))
            r2_rf  = r2_score(y_te, rf.predict(X_te))
            model = gbr if r2_gbr >= r2_rf else rf
        bucket_models[role][b] = {'model': model, 'cols': in_cols}

for idx, row in df.iterrows():
    role = row['role']
    for b in buckets:
        mdl = bucket_models[role][b]['model']
        cols = bucket_models[role][b]['cols']
        X = row[cols].fillna(0).values.reshape(1,-1) if len(cols)>0 else np.zeros((1,1))
        pred = float(mdl.predict(X)[0])
        preds[b].append(np.clip(pred, 0, 100))

for b in buckets:
    df[f'S_{b}'] = preds[b]

df[['partner_id','role','S_P','S_I','S_T','S_R','S_B','S_L','S_P','S_I']].head()

role_weights = {
    'driver': {'w_P':0.18,'w_I':0.15,'w_T':0.15,'w_R':0.12,'w_B':0.25,'w_L':0.15},
    'merchant':{'w_P':0.20,'w_I':0.17,'w_T':0.12,'w_R':0.11,'w_B':0.20,'w_L':0.20}
}

def compute_CWS_r(row):
    rw = role_weights.get(row['role'])
    numerator = (
        rw['w_P'] * row['S_P'] +
        rw['w_I'] * row['S_I'] +
        rw['w_T'] * row['S_T'] +
        rw['w_R'] * row['S_R'] +
        rw['w_B'] * row['S_B'] +
        rw['w_L'] * row['S_L']
    )
    denom = sum(rw.values())
    return float(100.0 * (numerator / denom) / 100.0)  
df['CWS_r'] = df.apply(compute_CWS_r, axis=1)
df['CWS_r'] = clamp(df['CWS_r'], 0, 100)

logit = (-(df['S_P']*0.28 + df['S_I']*0.22 + df['S_T']*0.18 + df['S_R']*0.12 + df['S_B']*0.12 + df['S_L']*0.08) + 60) / 8.0
p_default = 1.0 / (1.0 + np.exp(-logit))  
df['default'] = np.random.binomial(1, p_default)

p_r_models = {}
for role in ['driver','merchant']:
    m = df['role']==role
    Xc = df.loc[m, ['CWS_r']].values
    y = 1 - df.loc[m, 'default'].values  
    if len(Xc) >= 10:
        lr = LogisticRegression(max_iter=400)
        lr.fit(Xc, y)
        p_r_models[role] = lr
    else:
        p_r_models[role] = None

df['p_r'] = 0.0
for role in ['driver','merchant']:
    m = df['role']==role
    if p_r_models[role] is not None:
        probs = p_r_models[role].predict_proba(df.loc[m, ['CWS_r']].values)[:,1]
    else:
        probs = np.full(m.sum(), 0.5)
    df.loc[m, 'p_r'] = probs

global_feats = ['S_P','S_I','S_T','S_R','S_B','S_L']
Xg = df[global_feats].values
yg = 1 - df['default'].values
lrg = LogisticRegression(max_iter=600)
lrg.fit(Xg, yg)
df['p_global'] = lrg.predict_proba(Xg)[:,1]


lambda_map = {}
for role in ['driver','merchant']:
    n = (df['role']==role).sum()
    lambda_map[role] = float(n) / float(n + 300.0)

df['lambda_r'] = df['role'].map(lambda_map)

adj_map = {'driver': 0.0, 'merchant': 0.0}
df['Adj_r'] = df['role'].map(adj_map)

df['FinalCWS_raw'] = 100.0 * (df['lambda_r'] * df['p_r'] + (1.0 - df['lambda_r']) * df['p_global']) + df['Adj_r']
df['FinalCWS'] = clamp(df['FinalCWS_raw'], 0, 100)

df[['partner_id','role','CWS_r','p_r','p_global','lambda_r','Adj_r','FinalCWS']].head()


BaseLimit = 50_000.0
alpha = 0.8
beta = 0.5

df['LoanLimit'] = (
    BaseLimit
    * (1.0 + alpha * (df['FinalCWS'] - 50.0) / 50.0)
    * (1.0 + beta * (df['S_I'] - 50.0) / 50.0)
)
df['LoanLimit'] = df['LoanLimit'].clip(1000, 500000)

BaseRate = 0.20
gamma = 0.10
delta = 0.05
df['InterestRate'] = (BaseRate - gamma * (df['FinalCWS'] / 100.0) - delta * (df['S_R'] / 100.0))
df['InterestRate'] = df['InterestRate'].clip(0.05, 0.30)

df[['partner_id','role','FinalCWS','LoanLimit','InterestRate']].head()


plt.figure()
plt.hist(df['FinalCWS'], bins=30)
plt.title('FinalCWS distribution')
plt.xlabel('FinalCWS'); plt.ylabel('count')
plt.show()

plt.figure()
for role, grp in df.groupby('role'):
    plt.plot(sorted(grp['FinalCWS']), label=role)
plt.title('Sorted FinalCWS by role (visual comparability)')
plt.xlabel('rank'); plt.ylabel('FinalCWS')
plt.legend()
plt.show()


out_csv = os.path.join(os.path.expanduser("~"), "hybrid_cws_results.csv")
cols_out = ['partner_id','role','FinalCWS','GlobalCWS' if 'GlobalCWS' in df.columns else 'CWS_r','CWS_r','S_P','S_I','S_T','S_R','S_B','S_L','LoanLimit','InterestRate']
cols_out = [c for c in cols_out if c in df.columns]
df[cols_out].to_csv(out_csv, index=False)
print("Saved results to:", out_csv)