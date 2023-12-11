import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# データを読み込みます。
df = pd.read_csv('SSDSE-C-2023.csv', header=None, encoding='cp932')

# 都道府県庁所在地の名前を保存します。
cities = df.iloc[3:, 2]

# 分析対象のデータを抽出します。3行目までのデータを削除し、1列目から3列目までのデータも削除します。
df = df.iloc[3:, 3:]

# データを標準化します。
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

# KMeans クラスタリングを行います。
kmeans = KMeans(n_clusters=3, random_state=0)  # クラスタ数は適切に調整してください。
kmeans.fit(df_scaled)

# 結果のクラスタ番号を都道府県庁所在地に付与します。
cities_clustered = pd.DataFrame({'City': cities, 'Cluster': kmeans.labels_})

# クラスタ番号ごとに都道府県庁所在地を表示します。
for i in range(kmeans.n_clusters):
    print(f'Cluster {i}:')
    print(cities_clustered[cities_clustered['Cluster'] == i]['City'])
    print('\n')
