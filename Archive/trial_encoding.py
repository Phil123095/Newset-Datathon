import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

train_data = pd.read_csv('../train_data.csv', sep=',')


def decode_stations(df, column_decode, encoder):
    df[column_decode] = encoder.inverse_transform(df[column_decode])
    return df


def encode_and_cluster_stations(df, mean_var_list, cluster_list, groupby_var, n_clusters):
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

    print(conversion_dict)

    df['DC'] = df[groupby_var].apply(lambda x: conversion_dict[x])

    return df


label_enc = LabelEncoder()

train_data['new_station_code'] = label_enc.fit_transform(train_data['station_code'])

print(train_data['new_station_code'])

new_data_table = train_data[['ofd_date', 'new_station_code']]

new_data_table = decode_stations(new_data_table, 'new_station_code', label_enc)

print(new_data_table)
