import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna
import time
import matplotlib.pyplot as plt
import preprocess
import shap

def rmse(label, pred):
    return np.sqrt(mean_squared_error(label, pred))


def objective(trial):
    # パラメータの探索範囲
    params_tuning = {'min_sum_hessian_in_leaf': trial.suggest_loguniform('min_sum_hessian_in_leaf', 0.1, 10),
                     'max_depth': trial.suggest_int('max_depth', 3, 9),
                     'num_leaves': trial.suggest_int('num_leaves', 8, 512),
                     'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 0.95),
                     'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 0.95),
                     'num_iterations': trial.suggest_int('num_iterations', 10, 100)}

    # 学習
    params.update(params_tuning)
    model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_valid], valid_names=['train', 'valid'])

    # 予測
    va_pred = model.predict(va_x)

    # [0, 20] にクリップ(Overview -> Evaluation を参照)
    va_pred = va_pred.clip(0, 20)

    return rmse(va_y, va_pred)


"""
データ読み込み
"""
# 前処理済みデータセットを読み込む
train = pd.read_csv('../preprocessed/train.csv')
test_x = pd.read_csv('../preprocessed/test_x.csv')

# 型の最適化
preprocess.downcast_dtypes(train)
preprocess.downcast_dtypes(test_x)

# 訓練データを特徴量と目的変数に分離
train_x = train.drop('target', axis=1)
train_y = train['target']

"""
パラメータチューニング
"""
# チューニングしないパラメータ
params = {'objective': 'regression',
          'learning_rate': 0.1,
          'lambda_l1': 0.0,
          'lambda_l2': 1.0,
          'seed': 0,
          'metric': 'rmse',
          'verbosity': -1}

# 処理時間表示用
start = time.time()

val_month = 33

# 2013.11~2014.10を訓練データに、2014.11をバリデーションデータとする
is_va = (train_x['date_block_num'] == val_month) \
        & train_x['shop_id'].isin(list(test_x['shop_id'])) \
        & train_x['item_id'].isin(list(test_x['item_id']))
is_tr = (~ is_va) & (train_x['date_block_num'] < val_month)

# 訓練データとバリデーションデータに分ける
tr_x, va_x = train_x[is_tr], train_x[is_va]
tr_y, va_y = train_y[is_tr], train_y[is_va]

# [0, 20] にクリップ(Overview -> Evaluation を参照)
tr_y, va_y = tr_y.clip(0, 20), va_y.clip(0, 20)

# 訓練データとバリデーションデータを学習に使える形に変換
lgb_train = lgb.Dataset(tr_x, tr_y)
lgb_valid = lgb.Dataset(va_x, va_y)

# パラメータチューニング
study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
study.optimize(objective, timeout=600)

# 最適なパラメータを保存
params.update(study.best_params)

"""
テストデータの予測
"""
# 訓練データを学習に使える形に変換
lgb_train = lgb.Dataset(train_x, train_y)

# 学習
model = lgb.train(params, lgb_train)

# 予測
pred = model.predict(test_x.drop('ID', axis=1))

print("予測完了")


"""
提出ファイル作成
"""
# [0, 20] にクリップ(Overview -> Evaluation を参照)
submission = pd.concat([test_x['ID'], pd.Series(pred.clip(0, 20), name='item_cnt_month')], axis=1)

# 提出用ファイル作成
submission.to_csv('../submission/submission.csv', index=False)

"""
可視化
"""
# Feature Importances
# lgb.plot_importance(model, height=0.5, figsize=(8, 16), importance_type='gain')
# plt.savefig('../visual/importance_gain.png')
# lgb.plot_importance(model, height=0.5, figsize=(8, 16), importance_type='split')
# plt.savefig('../visual/importance_split.png')

# SHAP
explainer = shap.TreeExplainer(model=model)
shap_values = explainer.shap_values(X=test_x.drop('ID', axis=1))
shap.summary_plot(shap_values, test_x.drop('ID', axis=1), show=False)
plt.savefig('../visual/summary_plot.png', bbox_inches="tight")

print("predict_time:{0}".format(time.time() - start) + "[sec]")
