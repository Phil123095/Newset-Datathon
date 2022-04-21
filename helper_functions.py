import pandas as pd
import numpy as np
from functools import partial
from scipy.stats.mstats import winsorize
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.cluster import KMeans
import xgboost as xgb


# Change Date Vars based on column names
def date_vars_create(df, base_date_column, columns_to_create):
    df[base_date_column] = pd.to_datetime(df[base_date_column], format='%Y-%m-%d')
    df.sort_values(by=[base_date_column, 'station_code'], inplace=True)

    for column in columns_to_create:
        if 'weekday' == column:
            df[column] = df[base_date_column].dt.day_name()
        if 'month' == column:
            df[column] = df[base_date_column].dt.month_name()
        if 'day_of_month' == column:
            df[column] = df[base_date_column].dt.day

    return df


def encode_and_cluster_stations(df, mean_var_list, cluster_list, groupby_var, n_clusters, training_only=True):
    # Encode based on two means
    df_grouped = pd.DataFrame(df.groupby(by=[groupby_var], as_index=False).mean())
    # df_grouped = df_grouped.drop(columns = ['country_code'])

    count = 0
    for variable in mean_var_list:
        count += 1
        mean_var = df_grouped.groupby(groupby_var)[variable].mean()
        df_grouped.loc[:, ('en_var' + str(count))] = df_grouped[groupby_var].map(mean_var)

    df_grouped['mean_encode'] = df_grouped[mean_var_list[0]] + df_grouped[mean_var_list[1]]

    # Create Cluster
    df_cluster = df_grouped[cluster_list + ['mean_encode']]

    kmean = KMeans(n_clusters=n_clusters, random_state=0).fit(df_cluster)

    # Map DC to New Clusters
    df_mapping = list(kmean.predict(df_cluster))
    conversion_dict = {}

    for i in range(0, len(df_mapping)):
        conversion_dict[df_grouped[groupby_var][i]] = df_mapping[i]

    df['DC'] = df[groupby_var].apply(lambda x: conversion_dict[x])

    if training_only:
        return df

    else:
        return conversion_dict


def vars_encode(df, var_list):
    if 'fc_codes' in var_list:
        var_list = [x for x in var_list if x != 'fc_codes']
        df = encode_fc(df, 'fc_codes')

    prefix_list = ['DoM' if x == 'day_of_month' else x for x in var_list]
    new_data = pd.get_dummies(df, columns=var_list, prefix=prefix_list, prefix_sep='_')

    return new_data


def encode_fc(df, col_name):
    vals = list(df[col_name].str.split(', ').values)
    vals = [i for l in vals for i in l]
    vals = list(set(vals))
    vals.sort()

    for v in vals:
        n = col_name + '_' + v
        df[n] = df[col_name].str.contains(v)
        df[n] = df[n].astype('uint8')
    df.drop(columns=[col_name], inplace=True)
    return df


def outlier_management(df, column_list, perc_limit=0.025):
    for column in column_list:
        df[column] = winsorize(df[column], limits=(perc_limit, perc_limit))
    return df


# Creating Lags

def addSimpleLags_Diffs(df, lag_list, col_list, change_choice='lag'):
    if change_choice == 'lead':
        lag_list = map(lambda x: x * (-1), lag_list)

    arr_lags = list(map(partial(_buildLags_Diffs, df=df,
                                col_list=col_list,
                                change_choice=change_choice),
                        lag_list))

    df = pd.concat([df] + arr_lags, axis=1)

    return df


def _buildLags_Diffs(lag, df, col_list, change_choice):
    if change_choice == 'lag' or change_choice == 'lead':
        return df.groupby('station_code')[col_list].shift(lag).add_suffix(f'_{np.abs(lag)}_{change_choice}')
    elif change_choice == 'diff':
        return df.groupby('station_code')[col_list].diff(lag).add_suffix(f'_{np.abs(lag)}_{change_choice}')


def split_function(df, target_var, train_perc=.8):
    all_dates_unique = df['ofd_date'].unique()

    cutoff_date = all_dates_unique[int(len(all_dates_unique) * train_perc)]

    train_data = df[df.ofd_date < cutoff_date]
    test_data = df[df.ofd_date >= cutoff_date]

    train_data.set_index(['ofd_date'], inplace=True)
    test_data.set_index(['ofd_date'], inplace=True)

    train_X, train_Y = train_data.drop(target_var, axis=1), train_data[target_var]
    test_X, test_Y = test_data.drop(target_var, axis=1), test_data[target_var]

    return train_X, train_Y, test_X, test_Y


def split_function_large(df, target_var, train_perc1=.7, train_perc2=.2):
    all_dates_unique = df['ofd_date'].unique()

    cutoff_date_one = all_dates_unique[int(len(all_dates_unique) * train_perc1)]
    cutoff_date_two = all_dates_unique[int(len(all_dates_unique) * (train_perc1 + train_perc2))]

    train_data_one = df[df.ofd_date < cutoff_date_one]
    mask = ((df['ofd_date'] >= cutoff_date_one) & (df['ofd_date'] < cutoff_date_two))
    train_data_two = df.loc[mask]
    test_data = df[df.ofd_date >= cutoff_date_two]

    train_data_one.set_index(['ofd_date'], inplace=True)
    train_data_two.set_index(['ofd_date'], inplace=True)
    test_data.set_index(['ofd_date'], inplace=True)

    train_X_one, train_Y_one = train_data_one.drop(target_var, axis=1), train_data_one[target_var]
    train_X_two, train_Y_two = train_data_two.drop(target_var, axis=1), train_data_two[target_var]
    test_X, test_Y = test_data.drop(target_var, axis=1), test_data[target_var]

    return train_X_one, train_Y_one, train_X_two, train_Y_two, test_X, test_Y


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
        selection_model = xgb.XGBRegressor(n_estimators=250)
        selection_model.fit(select_X_train_one, y_train_one,
                            eval_set=[(select_X_train_one, y_train_one), (select_X_train_two, y_train_two)],
                            early_stopping_rounds=20,
                            verbose=False)

        preds2 = pd.DataFrame(selection_model.predict(select_X_test))
        RMSE = mean_squared_error(y_test, preds2, squared=False)
        all_RMSE.append(RMSE)

    average_RMSE = sum(all_RMSE) / len(all_RMSE)
    return average_RMSE


def feature_ranker(x_train_one, y_train_one, x_train_two, y_train_two, x_test, y_test, target_var):
    xgb_reg = xgb.XGBRegressor(n_estimators=1000)
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
        if 'station_code' not in features_to_use:
            features_to_use = features_to_use + ['station_code']
        select_X_train_one = x_train_one[features_to_use]
        select_X_train_two = x_train_two[features_to_use]
        select_X_test = x_test[features_to_use]

        avg_RMSE = cross_val_test(1, select_X_train_one=select_X_train_one, y_train_one=y_train_one,
                                  select_X_train_two=select_X_train_two, y_train_two=y_train_two,
                                  select_X_testBx=select_X_test, y_test=y_test)

        results[avg_RMSE] = features_to_use
        print(
            f"Feature Sel. RMSE for {target_var}, Score Threshold: {score}, Total features: {len(features_to_use)}, Average RMSE: {avg_RMSE}")

    lowest_RMSE = min([x for x in results.keys()])
    optimal_params = results[lowest_RMSE]
    print(f"Lowest RMSE: {lowest_RMSE} vs. Baseline: {RMSE_baseline}, with {len(optimal_params)} features: {optimal_params} ")
    return optimal_params
