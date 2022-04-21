import warnings

warnings.filterwarnings("ignore")

from data_processing import data_processor_training, data_processor_full
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


def convert_to_final(test_X, preds, var):
    colname = 'yhat_' + str(var)
    preds[colname] = preds[0]
    test_X = test_X.reset_index()

    full_prediction = test_X.join(preds)
    full_prediction = full_prediction[['ofd_date', 'station_code', colname]]
    return full_prediction


def model_manager(target_var, train_data, train_data_targets, test_data, params, scoring_func, MODELS_SET, PARAM_SEL):
    train_data.set_index(['ofd_date'], inplace=True)
    train_data_targets.set_index(['ofd_date'], inplace=True)
    test_data.set_index(['ofd_date'], inplace=True)

    print(f"Working on: {target_var}")
    final_train_target = train_data_targets[[target_var]]

    if MODELS_SET:
        if PARAM_SEL:
            prediction_outcome = xGB_SET_FEATURE_PRED(x_train=train_data, y_train=final_train_target, x_test=test_data,
                                         target_var=target_var, scoring_func=scoring_func)

        else:
            prediction_outcome = xGB_SET(x_train=train_data, y_train=final_train_target, x_test=test_data,
                                         target_var=target_var)
        print(f"{target_var} done")
        return prediction_outcome

    else:
        if PARAM_SEL:
            best_XGB, prediction_outcome = xGBGridSearch_Features(x_train=train_data, y_train=final_train_target,
                                                        x_test=test_data, params=params, scoring_func=scoring_func,
                                                        target_var=target_var)
        else:
            best_XGB, prediction_outcome = xGBGrid_FIND(x_train=train_data, y_train=final_train_target,
                                                        x_test=test_data, params=params, scoring_func=scoring_func,
                                                        target_var=target_var)
        print(f"{target_var} done")
        return best_XGB, prediction_outcome


def xGBGrid_FIND(x_train, y_train, x_test, params, scoring_func, target_var):
    xgb_reg = xgb.XGBRegressor(n_estimators=200)
    grid = HalvingGridSearchCV(xgb_reg, param_grid=params, factor=4, scoring=scoring_func, n_jobs=-1, verbose=3)
    # grid = GridSearchCV(xgb_reg, params, scoring=scoring_func, n_jobs=-1, cv=5, verbose=3)
    grid.fit(x_train, y_train)
    gridcv_xgb = grid.best_estimator_
    print(f"Best Params: {gridcv_xgb}")
    preds = gridcv_xgb.predict(x_test)
    preds_df = pd.DataFrame(preds)

    final_pred = convert_to_final(x_test, preds_df, target_var)
    return gridcv_xgb, final_pred


def xGBGridSearch_Features(x_train, y_train, x_test, params, scoring_func, target_var):
    x_train_transformed, x_test_transformed = xGB_SET_FEATURE(x_train=x_train, y_train=y_train, scoring_func=scoring_func, x_test=x_test)
    csv_name = 'just_to_see' + target_var + '.csv'
    x_train_transformed.to_csv(csv_name)
    xgb_reg = xgb.XGBRegressor(n_estimators=200)
    grid = HalvingGridSearchCV(xgb_reg, param_grid=params, factor=4, scoring=scoring_func, cv=5, n_jobs=-1, verbose=3)
    grid.fit(x_train_transformed, y_train)
    gridcv_xgb = grid.best_estimator_
    print(f"Best Params: {gridcv_xgb}")
    preds = gridcv_xgb.predict(x_test_transformed)
    preds_df = pd.DataFrame(preds)

    final_pred = convert_to_final(x_test, preds_df, target_var)
    return gridcv_xgb, final_pred

def xGB_provider(target_var, n_estimators):
    if target_var == 'MNR_SNR_Exp':
        xgb_reg = xgb.XGBRegressor(n_estimators=n_estimators, colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                                   gamma=0.4, learning_rate=0.3, max_depth=4, min_child_weight=4,
                                   subsample=1)
    elif target_var == 'Earlies_Exp':
        xgb_reg = xgb.XGBRegressor(n_estimators=n_estimators, colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.9,
                                   gamma=0.4, learning_rate=0.3, max_depth=5, min_child_weight=5,
                                   subsample=0.9)
    return xgb_reg


def xGB_SET(x_train, y_train, x_test, target_var):
    xgb_model = xGB_provider(target_var, n_estimators=1000)
    xgb_model.fit(x_train, y_train)

    preds = xgb_model.predict(x_test)
    preds_df = pd.DataFrame(preds)

    final_pred = convert_to_final(x_test, preds_df, target_var)
    return final_pred

def xGB_SET_FEATURE_PRED(x_train, y_train, target_var, scoring_func, x_test=None):
    xgb_model = xGB_provider(target_var, n_estimators=100)
    selector = RFECV(xgb_model, step=15, min_features_to_select=10, cv=4, scoring=scoring_func, n_jobs=-1, verbose=3)
    selector.fit(x_train, y_train)

    preds = selector.predict(x_test)
    preds_df = pd.DataFrame(preds)

    final_pred = convert_to_final(x_test, preds_df, target_var)
    return final_pred

def xGB_SET_FEATURE(x_train, y_train, scoring_func, x_test):
    xgb_reg = xgb.XGBRegressor(n_estimators=200)
    selector = RFECV(xgb_reg, step=8, min_features_to_select=1, cv=5, scoring=scoring_func, n_jobs=-1, verbose=3)
    selector.fit(x_train, y_train)

    features = selector.get_support(1)
    x_train_small = x_train[x_train.columns[features]]
    x_test_small = x_test[x_test.columns[features]]

    print(x_train_small)

    return x_train_small, x_test_small


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def final_merger(mnr_df, earlies_df, label_enc):
    mnr_df.reset_index(drop=True, inplace=True)
    earlies_df.reset_index(drop=True, inplace=True)

    forecast_full = pd.merge(mnr_df, earlies_df, how="left", on=["ofd_date", "station_code"])
    forecast_full['Expected'] = forecast_full['yhat_Earlies_Exp'] - forecast_full['yhat_MNR_SNR_Exp']

    forecast_full['station_code'] = label_enc.inverse_transform(forecast_full['station_code'])

    forecast_full['Id'] = forecast_full['ofd_date'].astype(str) + "_" + forecast_full['station_code']

    forecast_final_result = forecast_full[['Id', 'Expected']]
    return forecast_final_result


if __name__ == '__main__':
    train_data = pd.read_csv('./train_data.csv', sep=',')
    test_data = pd.read_csv('./test.csv', sep=',')

    target_vars = [['MNR_SNR_Exp', 'Earlies_Exp'], ['Earlies_Exp', 'MNR_SNR_Exp']]
    target_vars_simple = ['Earlies_Exp', 'MNR_SNR_Exp']
    base_date_col = 'ofd_date'
    date_columns_to_create = ['weekday', 'month', 'day_of_month']

    lag_to_do = []
    for x in range(1, 31):
        lag_to_do.append(x)

    diff_to_do = [1, 2, 3, 7]

    xGparams_high = {
        'min_child_weight': [3, 4, 5, 6],
        'gamma': [i / 10.0 for i in range(4, 6)],
        'subsample': [i / 10.0 for i in range(7, 11)],
        'colsample_bytree': [i / 10.0 for i in range(7, 11)],
        'max_depth': [4, 5, 6]
    }

    xGparams_medium = {
        'min_child_weight': [3, 4, 5],
        'gamma': [i / 10.0 for i in range(4, 6)],
        'subsample': [i / 10.0 for i in range(8, 11)],
        'colsample_bytree': [i / 10.0 for i in range(8, 11)],
        'max_depth': [3, 4, 5]
    }

    xGparams_low = {
        'min_child_weight': [4, 5],
        'gamma': [i / 10.0 for i in range(4, 6)],
        'subsample': [i / 10.0 for i in range(9, 11)],
        'colsample_bytree': [i / 10.0 for i in range(9, 11)],
        'max_depth': [4, 5]
    }

    # Run data processor
    results_dict = {}
    label_enc = LabelEncoder()
    mse_scorer = make_scorer(rmse, greater_is_better=False)

    result_dfs = []

    models_set = False
    for i in range(len(target_vars)):
        train_data = pd.read_csv('./train_data.csv', sep=',')
        test_data = pd.read_csv('./test.csv', sep=',')
        train_data_processed, train_target_vars, test_data_processed = data_processor_full(train_df=train_data,
                                                                                           target_df=test_data,
                                                                                           dep_vars=target_vars_simple,
                                                                                           date_col=base_date_col,
                                                                                           date_cols_ToDo=date_columns_to_create,
                                                                                           label_enc=label_enc,
                                                                                           lag_ToDo=lag_to_do,
                                                                                           diff_ToDo=diff_to_do,
                                                                                           mean_vars_clustering=['OFD',
                                                                                                                 'Slam'],
                                                                                           cluster_groupby='station_code',
                                                                                           n_clusters=7,
                                                                                           outlier_threshold=0.025,
                                                                                           lagChoice='both',
                                                                                           cutoff_date='2021-07-01',
                                                                                           verbose=False)

        if models_set:
            result_prediction = model_manager(target_var=target_vars[i][0],
                                              train_data=train_data_processed,
                                              train_data_targets=train_target_vars,
                                              test_data=test_data_processed,
                                              params=xGparams_low, scoring_func=mse_scorer,
                                              MODELS_SET=models_set,
                                              PARAM_SEL=True)



        else:
            best_params, result_prediction = model_manager(target_var=target_vars[i][0],
                                                           train_data=train_data_processed,
                                                           train_data_targets=train_target_vars,
                                                           test_data=test_data_processed,
                                                           params=xGparams_high, scoring_func=mse_scorer,
                                                           MODELS_SET=models_set,
                                                           PARAM_SEL=True)
            results_dict[target_vars[i][0]] = {}
            results_dict[target_vars[i][0]]['best_param'] = best_params

        result_dfs.append(result_prediction)

    final_output = final_merger(result_dfs[0], result_dfs[1], label_enc)
    print(final_output)
    final_output.to_csv('phil_submission_final.csv', index=False)
