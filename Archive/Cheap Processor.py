import warnings

warnings.filterwarnings("ignore")

from data_processing import data_processor_full, data_processor_full_jun
from Archive.helper_functions import split_function_large, feature_ranker, split_function
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import HalvingGridSearchCV


def convert_to_final(test_X, preds, var):
    colname = 'yhat_' + str(var)
    preds[colname] = preds[0]
    test_X = test_X.reset_index()

    full_prediction = test_X.join(preds)
    print(full_prediction)
    full_prediction = full_prediction[['ofd_date', 'station_code', colname]]
    return full_prediction


def model_manager_features(full_df_train, target_var, test_data, non_target_var, params):
    print(f"Working on: {target_var}")
    full_df_train.drop(columns=[non_target_var], axis=1, inplace=True)
    train_X_one, train_Y_one, train_X_two, train_Y_two, test_X, test_Y = split_function_large(df=full_df_train,
                                                                                              target_var=target_var,
                                                                                              train_perc1=0.7,
                                                                                              train_perc2=0.2)

    optimal_features = feature_ranker(x_train_one=train_X_one, y_train_one=train_Y_one, x_train_two=train_X_two,
                                      y_train_two=train_Y_two, x_test=test_X, y_test=test_Y, target_var=target_var)

    print(['ofd_date'] + optimal_features)

    full_train_df_selected = full_df_train[['ofd_date'] + optimal_features + [target_var]]
    print(full_train_df_selected)
    full_test_selected = test_data[['ofd_date'] + optimal_features]
    print(full_test_selected)

    train_X, train_Y, validation_X, validation_Y = split_function(df=full_train_df_selected,
                                                                  target_var=target_var,
                                                                  train_perc=0.8)

    optimal_params, final_prediction = xGBGrid_FIND(x_train=train_X, y_train=train_Y, x_validate=validation_X,
                                                    y_validate=validation_Y, x_test=full_test_selected,
                                                    params=params, target_var=target_var)

    return optimal_features, optimal_params, final_prediction


def xGBGrid_FIND(x_train, y_train, x_validate, y_validate, x_test, params, target_var):
    xgb_reg = xgb.XGBRegressor()
    grid = HalvingGridSearchCV(xgb_reg, param_grid=params, factor=3, scoring='neg_mean_squared_error', n_jobs=250,
                               verbose=2)
    # grid = GridSearchCV(xgb_reg, params, scoring=scoring_func, n_jobs=-1, cv=5, verbose=3)
    grid.fit(x_train, y_train, eval_set=[(x_validate, y_validate)], early_stopping_rounds=20, verbose=False)
    gridcv_xgb = grid.best_estimator_
    print(f"Best Params: {gridcv_xgb}")


    x_base = x_test[['ofd_date', 'station_code']]

    x_test = x_test[list(x_train.columns)]
    preds = gridcv_xgb.predict(x_test)
    preds_df = pd.DataFrame(preds)

    final_pred = convert_to_final(x_base, preds_df, target_var)
    return gridcv_xgb, final_pred


def final_merger(mnr_df, earlies_df):
    mnr_df.reset_index(drop=True, inplace=True)
    earlies_df.reset_index(drop=True, inplace=True)

    forecast_full = pd.merge(mnr_df, earlies_df, how="left", on=["ofd_date", "station_code"])
    forecast_full['Expected'] = forecast_full['yhat_Earlies_Exp'] - forecast_full['yhat_MNR_SNR_Exp']

    forecast_full['Id'] = forecast_full['ofd_date'].astype(str) + "_D" + forecast_full['station_code'].astype(str)

    forecast_final_result = forecast_full[['Id', 'Expected']]
    return forecast_final_result


def final_merger_Y(Y_df, label_enc):

    Y_df['Expected'] = Y_df['Earlies_Exp'] - Y_df['MNR_SNR_Exp']

    Y_df['station_code'] = label_enc.inverse_transform(Y_df['station_code'])
    Y_df['Id'] = Y_df['ofd_date'].astype(str) + "_" + Y_df['station_code']

    test_final_result = Y_df[['Id', 'Expected']]
    return test_final_result


if __name__ == '__main__':

    target_vars = [['MNR_SNR_Exp', 'Earlies_Exp'], ['Earlies_Exp', 'MNR_SNR_Exp']]
    target_vars_simple = ['Earlies_Exp', 'MNR_SNR_Exp']
    base_date_col = 'ofd_date'
    date_columns_to_create = ['weekday', 'month', 'day_of_month']

    lag_to_do = []
    for x in range(1, 31):
        lag_to_do.append(x)


    diff_to_do = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

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
        'learning_rate': [0.05, 0.1, 0.2],
        'n_estimators': [500],
        'min_child_weight': [4, 5],
        'gamma': [i / 10.0 for i in range(4, 6)],
        'subsample': [i / 10.0 for i in range(8, 10)],
        'colsample_bytree': [i / 10.0 for i in range(8, 11)],
        'max_depth': [3, 4, 5]
    }

    # Run data processor
    results_dict = {}

    result_dfs = []

    synth_tests = []

    early_testing = False

    for i in range(len(target_vars)):
        train_data = pd.read_csv('../train_data.csv', sep=',')
        test_data = pd.read_csv('../test.csv', sep=',')

        if early_testing:
            train_data_processed, final_all_train, train_target_vars, test_data_processed, synth_test_Y = data_processor_full_jun(
                train_df=train_data,
                dep_vars=target_vars_simple,
                date_col=base_date_col,
                date_cols_ToDo=date_columns_to_create,
                lag_ToDo=lag_to_do,
                diff_ToDo=diff_to_do,
                outlier_threshold=0.025,
                lagChoice='both',
                cutoff_date='2021-06-01',
                verbose=False)

            print(final_all_train.columns)

            synth_tests.append(synth_test_Y)

            optimal_features, optimal_params, result_prediction = model_manager_features(full_df_train=final_all_train,
                                                                                         target_var=target_vars[i][0],
                                                                                         test_data=test_data_processed,
                                                                                         non_target_var=target_vars[i][
                                                                                             1], params=xGparams_low)



            results_dict[target_vars[i][0]] = {}
            results_dict[target_vars[i][0]]['best_features'] = optimal_features
            results_dict[target_vars[i][0]]['best_param'] = optimal_params

            result_dfs.append(result_prediction)

        else:
            final_all_train, test_data_processed = data_processor_full(
                train_df=train_data,
                target_df=test_data,
                dep_vars=target_vars_simple,
                date_col=base_date_col,
                date_cols_ToDo=date_columns_to_create,
                lag_ToDo=lag_to_do,
                diff_ToDo=diff_to_do,
                outlier_threshold=0.025,
                lagChoice='both',
                cutoff_date='2021-07-01',
                verbose=False)

            optimal_features, optimal_params, result_prediction = model_manager_features(full_df_train=final_all_train,
                                                                                         target_var=target_vars[i][0],
                                                                                         test_data=test_data_processed,
                                                                                         non_target_var=target_vars[i][
                                                                                             1], params=xGparams_low)



            results_dict[target_vars[i][0]] = {}
            results_dict[target_vars[i][0]]['best_features'] = optimal_features
            results_dict[target_vars[i][0]]['best_param'] = optimal_params

            result_dfs.append(result_prediction)

    final_output = final_merger(result_dfs[0], result_dfs[1])
    #final_test_Y = final_merger_Y(synth_tests[0], label_enc)

    #final_output.set_index('Id', inplace=True)
    #final_test_Y.set_index('Id', inplace=True)

    #final_RMSE = mean_squared_error(final_test_Y, final_output, squared=False)
    #print(f"Final RMSE on June: {final_RMSE}")

    print(final_output)
    final_output.to_csv('phil_submission_final_squared.csv', index=False)
