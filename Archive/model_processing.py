import warnings

warnings.filterwarnings("ignore")
from data_processing import data_processor_training
from helper_functions import split_function, split_function_large
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


def model_manager(full_df, target_var, non_target_var, params, scoring_func):
    fc_drops = [x for x in list(full_df.columns) if 'fc_codes' in x]
    full_df.drop(columns=fc_drops, inplace=True)
    print(f"Working on: {target_var}")
    full_df.drop(columns=[non_target_var], axis=1, inplace=True)
    train_X, train_Y, test_X, test_Y = split_function(df=full_df, target_var=target_var, train_perc=0.8)
    best_XGB, result_RMSE, prediction_outcome = xGBGrid(x_train=train_X, y_train=train_Y, x_test=test_X,
                                                        y_test=test_Y, params=params, scoring_func=scoring_func,
                                                        target_var=target_var)
    print(f"{target_var} done")
    return best_XGB, result_RMSE, prediction_outcome


def model_manager_features(full_df, target_var, non_target_var):
    print(f"Working on: {target_var}")
    full_df.drop(columns=[non_target_var], axis=1, inplace=True)
    train_X_one, train_Y_one, train_X_two, train_Y_two, test_X, test_Y = split_function_large(df=full_df,
                                                                                              target_var=target_var,
                                                                                              train_perc1=0.7,
                                                                                              train_perc2=0.2)
    feature_ranker(x_train_one=train_X_one, y_train_one=train_Y_one, x_train_two=train_X_two,
                   y_train_two=train_Y_two, x_test=test_X, y_test=test_Y, target_var=target_var)


def convert_to_final(test_X, preds, var):
    colname = 'yhat_' + str(var)
    preds[colname] = preds[0]
    test_X = test_X.reset_index()

    full_prediction = test_X.join(preds)
    full_prediction = full_prediction[['ofd_date', 'station_code', colname]]
    return full_prediction


def return_columns(dict, threshold):
    column_list = []
    for feature in dict.keys():
        if dict[feature] >= threshold:
            column_list.append(feature)
        else:
            continue
    return column_list


def cross_val_test(nr_cross, select_X_train_one, y_train_one, select_X_train_two, y_train_two, select_X_test, y_test):
    all_RMSE = []
    for _ in range(nr_cross):
        selection_model = xgb.XGBRegressor(n_estimators=1000)
        selection_model.fit(select_X_train_one, y_train_one,
                            eval_set=[(select_X_train_one, y_train_one), (select_X_train_two, y_train_two)],
                            early_stopping_rounds=50,
                            verbose=False)

        preds2 = pd.DataFrame(selection_model.predict(select_X_test))
        RMSE = mean_squared_error(y_test, preds2, squared=False)
        all_RMSE.append(RMSE)

    average_RMSE = sum(all_RMSE) / len(all_RMSE)
    return average_RMSE


def feature_ranker(x_train_one, y_train_one, x_train_two, y_train_two, x_test, y_test, target_var):
    xgb_reg = xgb.XGBRegressor(n_estimators=500)
    xgb_reg.fit(x_train_one, y_train_one, eval_set=[(x_train_one, y_train_one), (x_train_two, y_train_two)],
                early_stopping_rounds=50,
                verbose=False)

    importances = xgb_reg.get_booster().get_score(importance_type='weight')

    importances_scores = [x for x in importances.values()]

    new_scores = []
    for score in importances_scores:
        if score not in new_scores:
            new_scores.append(score)

    new_scores.sort()

    preds = pd.DataFrame(xgb_reg.predict(x_test))
    RMSE_baseline = mean_squared_error(y_test, preds, squared=False)
    print(f"Base RMSE for {target_var}, All features: {RMSE_baseline}")

    results = {}
    for score in new_scores:
        features_to_use = return_columns(importances, score)

        print(features_to_use)
        select_X_train_one = x_train_one[features_to_use]
        select_X_train_two = x_train_two[features_to_use]
        select_X_test = x_test[features_to_use]

        avg_RMSE = cross_val_test(5, select_X_train_one=select_X_train_one, y_train_one=y_train_one,
                                  select_X_train_two=select_X_train_two, y_train_two=y_train_two,
                                  select_X_test=select_X_test, y_test=y_test)

        results[avg_RMSE] = features_to_use
        print(
            f"Feature Sel. RMSE for {target_var}, Score Threshold: {score}, Total features: {len(features_to_use)}, Average RMSE: {avg_RMSE}")

    lowest_RMSE = min([x for x in results.keys()])
    optimal_params = results[lowest_RMSE]
    print(f"Lowest RMSE: {lowest_RMSE} vs. Baseline: {RMSE_baseline}, with features: {optimal_params} ")


def xGBGrid(x_train, y_train, x_test, y_test, params, scoring_func, target_var):
    xgb_reg = xgb.XGBRegressor(n_estimators=100)
    grid = GridSearchCV(xgb_reg, params, n_jobs=-1, cv=3, verbose=3)
    grid.fit(x_train, y_train)
    gridcv_xgb = grid.best_estimator_
    print(f"Best Params: {gridcv_xgb}")
    preds = gridcv_xgb.predict(x_test)
    preds_df = pd.DataFrame(preds)
    RMSE = mean_squared_error(y_test, preds, squared=False)

    final_pred = convert_to_final(x_test, preds_df, target_var)
    print(f"RMSE: {RMSE}")
    return gridcv_xgb, RMSE, final_pred


def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


def final_merger(mnr_df, earlies_df, label_enc):
    mnr_df.reset_index(drop=True, inplace=True)
    earlies_df.reset_index(drop=True, inplace=True)

    forecast_full = pd.merge(mnr_df, earlies_df, how="left", on=["ofd_date", "station_code"])
    print(forecast_full)
    forecast_full['Expected'] = forecast_full['yhat_Earlies_Exp'] - forecast_full['yhat_MNR_SNR_Exp']

    forecast_full['station_code'] = label_enc.inverse_transform(forecast_full['station_code'])

    forecast_full['Id'] = forecast_full['ofd_date'].astype(str) + "_" + forecast_full['station_code']

    forecast_final_result = forecast_full[['Id', 'Expected']]
    return forecast_final_result


if __name__ == '__main__':
    # Define Base Parameters
    target_vars = [['MNR_SNR_Exp', 'Earlies_Exp'], ['Earlies_Exp', 'MNR_SNR_Exp']]
    target_vars_simple = ['Earlies_Exp', 'MNR_SNR_Exp']
    base_date_col = 'ofd_date'
    date_columns_to_create = ['weekday', 'month', 'day_of_month']

    feature_selection = True
    lag_to_do = []
    for x in range(1, 31):
        lag_to_do.append(x)

    diff_to_do = [1, 2, 3, 7]

    mse_scorer = make_scorer(rmse, greater_is_better=False)

    """
    xGparams = {
        'min_child_weight': [2, 3, 4, 5],
        'gamma': [i / 10.0 for i in range(3, 6)],
        'subsample': [i / 10.0 for i in range(6, 11)],
        'colsample_bytree': [i / 10.0 for i in range(6, 11)],
        'max_depth': [2, 3, 4, 5]
    }
    """

    xGparams = {
        'min_child_weight': [5],
        'gamma': [i / 10.0 for i in range(5, 6)],
        'subsample': [i / 10.0 for i in range(10, 11)],
        'colsample_bytree': [i / 10.0 for i in range(10, 11)],
        'max_depth': [4, 5]
    }

    # Run data processor
    results_dict = {}
    label_enc = LabelEncoder()

    result_dfs = []
    for i in range(len(target_vars)):
        train_data = pd.read_csv('../train_data.csv', sep=',')
        final_data = data_processor_training(df=train_data,
                                             dep_vars=target_vars_simple,
                                             date_col=base_date_col,
                                             date_cols_ToDo=date_columns_to_create,
                                             label_enc=label_enc,
                                             univariate_flag=False,
                                             lag_ToDo=lag_to_do,
                                             diff_ToDo=diff_to_do,
                                             mean_vars_clustering=['OFD', 'Slam'],
                                             cluster_groupby='station_code',
                                             n_clusters=7,
                                             outlier_threshold=0.025,
                                             lagChoice='both',
                                             verbose=False)

        if feature_selection:
            model_manager_features(full_df=final_data, target_var=target_vars[i][0],
                                   non_target_var=target_vars[i][1])

        else:

            best_params, result_RMSE, result_prediction = model_manager(full_df=final_data,
                                                                        target_var=target_vars[i][0],
                                                                        non_target_var=target_vars[i][1],
                                                                        params=xGparams, scoring_func=mse_scorer)
            result_dfs.append(result_prediction)

            print(result_prediction)

            results_dict[target_vars[i][0]] = {}
            results_dict[target_vars[i][0]]['best_param'] = best_params
            results_dict[target_vars[i][0]]['RMSE'] = result_RMSE

    if not feature_selection:
        final_output = final_merger(result_dfs[0], result_dfs[1], label_enc)
        print(results_dict)
        print(final_output)
