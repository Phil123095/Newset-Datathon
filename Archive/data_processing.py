from Archive.helper_functions import date_vars_create, encode_and_cluster_stations, vars_encode, outlier_management, \
    addSimpleLags_Diffs

import pandas as pd


def data_processor_training(df, dep_vars, date_col, date_cols_ToDo, label_enc, univariate_flag=False,
                            lag_ToDo=[1, 2, 4, 5, 6, 7],
                            diff_ToDo=None, mean_vars_clustering=['OFD', 'Slam'], cluster_groupby='station_code',
                            n_clusters=7,
                            outlier_threshold=0.025, lagChoice='both', verbose=False):
    if univariate_flag:
        df['Target'] = df['Earlies_Exp'] - df['MNR_SNR_Exp']
        # All int64, including target variables.
        LagDiff_vars_ToDo = [x for x in list(df.select_dtypes(['int64']).columns)]

    elif not univariate_flag:
        LagDiff_vars_ToDo = [x for x in list(df.select_dtypes(['int64']).columns) if x not in dep_vars]

    # All 'object' dtype columns from the original DF (excluding the date column) should be turned into dummies
    encode_basevars_ToDo = [x for x in list(df.select_dtypes(['object']).columns) if
                            x != date_col and x != 'station_code']
    encode_basevars_ToDo.append('DC')

    # Manage outliers
    df_mod = outlier_management(df, LagDiff_vars_ToDo, outlier_threshold)

    final_encode_list = encode_basevars_ToDo + date_cols_ToDo

    # Create Date Variables
    df_mod = date_vars_create(df_mod, date_col, date_cols_ToDo)

    if lagChoice == 'both':
        df_mod = addSimpleLags_Diffs(df_mod, lag_ToDo, LagDiff_vars_ToDo, change_choice='lag')
        df_mod = addSimpleLags_Diffs(df_mod, diff_ToDo, LagDiff_vars_ToDo, change_choice='diff')

    elif lagChoice in ['lag', 'diff', 'lead']:
        df_mod = addSimpleLags_Diffs(df_mod, lag_ToDo, LagDiff_vars_ToDo, change_choice=lagChoice)

    df_mod = df_mod.fillna(0)

    df_mod = encode_and_cluster_stations(df_mod, mean_var_list=mean_vars_clustering,
                                         cluster_list=LagDiff_vars_ToDo + dep_vars, groupby_var=cluster_groupby,
                                         n_clusters=n_clusters)
    # Encode columns. DC should be excluded from this as we do a special encoding.
    df_mod = vars_encode(df_mod, final_encode_list)
    df_mod['station_code'] = label_enc.fit_transform(df_mod['station_code'])

    if verbose:
        for column in list(df_mod.columns):
            print(column)
    else:
        print("Data processing is done.")
        print(f"We now have {len(list(df_mod.columns))} features. Woop woop.")

    return df_mod


def data_processor_full_jun(train_df, dep_vars, date_col, date_cols_ToDo, label_enc,
                        lag_ToDo=[1, 2, 4, 5, 6, 7],
                        diff_ToDo=None,
                        outlier_threshold=0.025, lagChoice='both', cutoff_date = '2021-07-01', verbose=False):

    train_df.sort_values(by=date_col, ascending=True, inplace=True)

    LagDiff_vars_ToDo = [x for x in list(train_df.select_dtypes(['int64']).columns) if x not in dep_vars]
    encode_basevars_ToDo = [x for x in list(train_df.select_dtypes(['object']).columns) if
                            x != date_col and x != 'station_code']
    final_encode_list = encode_basevars_ToDo + date_cols_ToDo

    train_df = outlier_management(train_df, LagDiff_vars_ToDo, outlier_threshold)

    full_data_to_process = date_vars_create(train_df, date_col, date_cols_ToDo)

    if lagChoice == 'both':
        full_data_to_process = addSimpleLags_Diffs(full_data_to_process, lag_ToDo, LagDiff_vars_ToDo, change_choice='lag')
        full_data_to_process = addSimpleLags_Diffs(full_data_to_process, diff_ToDo, LagDiff_vars_ToDo, change_choice='diff')

    elif lagChoice in ['lag', 'diff', 'lead']:
        full_data_to_process = addSimpleLags_Diffs(full_data_to_process, lag_ToDo, LagDiff_vars_ToDo, change_choice=lagChoice)

    full_data_to_process = full_data_to_process.fillna(0)

    # Encode columns. DC should be excluded from this as we do a special encoding.
    full_data_to_process = vars_encode(full_data_to_process, final_encode_list)
    full_data_to_process['station_code'] = label_enc.fit_transform(full_data_to_process['station_code'])

    train_target_vars = full_data_to_process[['ofd_date', 'station_code'] + dep_vars]
    train_df.drop(columns=dep_vars, inplace=True)


    train_data_processed = full_data_to_process[full_data_to_process.ofd_date < cutoff_date]
    train_data_processed = train_data_processed[train_data_processed.ofd_date > '2021-03-02']

    mask = ((train_target_vars.ofd_date > '2021-03-02') & (train_target_vars.ofd_date < cutoff_date))
    train_Y = train_target_vars.loc[mask]
    synth_test_data_Y = train_target_vars[train_target_vars.ofd_date >= cutoff_date]


    test_data_processed = full_data_to_process[full_data_to_process.ofd_date >= cutoff_date]

    train_Y['ofd_date'] = train_Y.ofd_date.astype('datetime64[ns]')
    train_data_processed['ofd_date'] = train_data_processed.ofd_date.astype('datetime64[ns]')
    train_Y['station_code'] = train_Y.station_code.astype(int)
    train_data_processed['station_code'] = train_data_processed.station_code.astype(int)

    final_train_all = pd.merge(train_data_processed, train_Y, how="left",  on=['ofd_date', 'station_code', 'Earlies_Exp', 'MNR_SNR_Exp'])

    final_train_all.reset_index(drop=True, inplace=True)
    train_data_processed.reset_index(drop=True, inplace=True)
    train_Y.reset_index(drop=True, inplace=True)
    test_data_processed.reset_index(drop=True, inplace=True)

    if verbose:
        for column in list(full_data_to_process.columns):
            print(column)
    else:
        print("Data processing is done.")
        print(f"We now have {len(list(full_data_to_process.columns))} features. Woop woop.")

    if cutoff_date == '2021-07-01':
        return train_data_processed, final_train_all, train_Y, test_data_processed

    else:
        return train_data_processed, final_train_all, train_Y, test_data_processed, synth_test_data_Y




def data_processor_full(train_df, target_df, dep_vars, date_col, date_cols_ToDo,
                        lag_ToDo=[1, 2, 4, 5, 6, 7],
                        diff_ToDo=None,
                        outlier_threshold=0.025, lagChoice='both', cutoff_date = '2021-07-01', verbose=False):

    train_df.sort_values(by=date_col, ascending=True, inplace=True)
    target_df.sort_values(by=date_col, ascending=True, inplace=True)

    LagDiff_vars_ToDo = [x for x in list(train_df.select_dtypes(['int64']).columns) if x not in dep_vars]
    encode_basevars_ToDo = [x for x in list(train_df.select_dtypes(['object']).columns) if
                            x != date_col and x != 'station_code']
    final_encode_list = encode_basevars_ToDo + date_cols_ToDo

    train_df = outlier_management(train_df, LagDiff_vars_ToDo, outlier_threshold)
    target_df = outlier_management(target_df, LagDiff_vars_ToDo, outlier_threshold)

    target_df.drop(columns=target_df.columns[0], axis=1, inplace=True)

    train_target_vars = train_df[['ofd_date', 'station_code'] + dep_vars]
    train_df.drop(columns=dep_vars, inplace=True)

    full_data_to_process = pd.concat([train_df, target_df], ignore_index=True)



    if lagChoice == 'both':
        full_data_to_process = addSimpleLags_Diffs(full_data_to_process, lag_ToDo, LagDiff_vars_ToDo, change_choice='lag')
        full_data_to_process = addSimpleLags_Diffs(full_data_to_process, diff_ToDo, LagDiff_vars_ToDo, change_choice='diff')

    elif lagChoice in ['lag', 'diff', 'lead']:
        full_data_to_process = addSimpleLags_Diffs(full_data_to_process, lag_ToDo, LagDiff_vars_ToDo, change_choice=lagChoice)

    for column in list(full_data_to_process.select_dtypes(['int64']).columns):
        full_data_to_process[f"{column}_square"] = full_data_to_process[column].pow(2)

    #full_data_to_process = date_vars_create(full_data_to_process, date_col, date_cols_ToDo)

    full_data_to_process = full_data_to_process.fillna(0)

    # Encode columns. DC should be excluded from this as we do a special encoding.
    #full_data_to_process = vars_encode(full_data_to_process, final_encode_list)
    full_data_to_process.drop(columns=encode_basevars_ToDo, inplace=True)

    if verbose:
        for column in list(full_data_to_process.columns):
            print(column)
    else:
        print("Data processing is done.")
        print(f"We now have {len(list(full_data_to_process.columns))} features. Woop woop.")

    print(full_data_to_process)
    full_data_to_process.to_csv('just_to_check.csv')


    train_data_processed = full_data_to_process[full_data_to_process.ofd_date < cutoff_date]
    train_data_processed = train_data_processed[train_data_processed.ofd_date > '2021-03-02']
    train_Y = train_target_vars[train_target_vars.ofd_date > '2021-03-02']
    test_data_processed = full_data_to_process[full_data_to_process.ofd_date >= cutoff_date]


    train_Y['ofd_date'] = train_Y.ofd_date.astype('datetime64[ns]')
    train_data_processed['ofd_date'] = train_data_processed.ofd_date.astype('datetime64[ns]')
    train_Y['station_code'] = train_Y.station_code.astype(str)
    train_data_processed['station_code'] = train_data_processed.station_code.astype(str)

    final_train_all = pd.merge(train_data_processed, train_Y, on=["ofd_date", 'station_code'], how="left")

    final_train_all['station_code'] = final_train_all['station_code'].apply(lambda x: int(x[1:]))

    test_data_processed['station_code'] = test_data_processed['station_code'].apply(lambda x: int(x[1:]))



    print(final_train_all)

    #final_train_all = pd.merge(train_data_processed, train_Y, how="left",  on=['ofd_date', 'station_code'])

    final_train_all.reset_index(drop=True, inplace=True)
    train_data_processed.reset_index(drop=True, inplace=True)
    train_Y.reset_index(drop=True, inplace=True)
    test_data_processed.reset_index(drop=True, inplace=True)


    return final_train_all, test_data_processed
