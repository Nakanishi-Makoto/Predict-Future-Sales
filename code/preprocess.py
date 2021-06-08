import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import time
import copy

# データフレームのprint時に改行や列の省略がされないように設定
pd.set_option("display.width", 1000)
pd.set_option('display.max_columns', 50)


# メモリ節約のため型を最適化
def downcast_dtypes(df):
    float32_columns = ['item_price', 'item_price_median']
    uint32_columns = ['ID']
    uint16_columns = ['item_id', 'item_cnt_day']
    object_columns = ['item_category_name', 'item_name', 'date', 'shop_name']
    
    for column in df.columns:
        if df[column].isnull().any():
            continue

        elif column in float32_columns:
            df[column] = df[column].astype('float32')

        elif column in uint32_columns:
            df[column] = df[column].astype('uint32')

        elif column in uint16_columns:
            df[column] = df[column].astype('uint16')

        elif column not in float32_columns + uint32_columns + uint16_columns + object_columns:
            df[column] = df[column].astype('uint8')

    return df


# item_category_name から小分類を作る
def make_sub_category(category):
    # 文字列 category が " - "を含んでいるなら、
    if category.find(" - ") >= 0:
        # " - " の右側を返す
        return category.split(" - ")[1]
    # 文字列 category が " - "を含んでいないなら、
    else:
        # 空文字を返す
        return ""


# デバッグ用　処理したファイルを保存しておく
def debug_save_files(self):
    self.train.to_csv('../debug/train.csv', index=False)
    self.test_x.to_csv('../debug/test_x.csv', index=False)


# デバッグ用　処理したファイルを読み込む
def debug_load_files(self):
    self.train = pd.read_csv('../debug/train.csv')
    self.test_x = pd.read_csv('../debug/test_x.csv')


if __name__ == '__main__':
    """
    データ読み込み
    """
    # 処理時間表示用
    start = time.time()

    # 生データ
    item_categories = pd.read_csv('../input/item_categories.csv')
    items = pd.read_csv('../input/items.csv')
    sales_train = pd.read_csv('../input/sales_train.csv')
    shops = pd.read_csv('../input/shops.csv')
    test = pd.read_csv('../input/test.csv')

    # 型の最適化
    item_categories = downcast_dtypes(item_categories)
    items = downcast_dtypes(items)
    sales_train = downcast_dtypes(sales_train)
    shops = downcast_dtypes(shops)
    test = downcast_dtypes(test)

    # 前処理済みデータ
    train = pd.DataFrame()
    test_x = test.copy()

    """
    データのクレンジング
    """
    # 外れ値の削除
    sales_train = sales_train[sales_train.item_price < 100000]
    sales_train = sales_train[sales_train.item_cnt_day < 1000]

    # item_price が -1 のデータポイントと 月/店ID/商品ID が一致しているデータポイントの商品価格の中央値を計算
    median = sales_train[(sales_train.date_block_num == 4)
                         & (sales_train.shop_id == 32)
                         & (sales_train.item_id == 2973)
                         & (sales_train.item_price > 0)].item_price.median()

    # 計算した中央値を item_price に代入
    sales_train.loc[sales_train.item_price < 0, 'item_price'] = median

    #

    """
    訓練データの拡張
    """
    # 組(日, 店, 商品)に対して販売個数が 0 のデータポイントは訓練データに含まれていないので、
    # 訓練データを販売個数が 0 のデータポイントを含めたすべての組(日, 店, 商品)に拡張する

    # 訓練データに含まれる組(店, 商品)について、34か月分のデータを作成
    for month in range(34):
        comb = sales_train[['shop_id', 'item_id']]
        comb.insert(2, 'date_block_num', month)
        train = pd.concat([train, comb])

        comb = test[['shop_id', 'item_id']]
        comb.insert(2, 'date_block_num', month)
        train = pd.concat([train, comb])

    # 重複している行を削除
    train = train.drop_duplicates()

    # train に item_category_id を追加
    train = pd.merge(train, items[['item_id', 'item_category_id']], how='left')

    """
    特徴量の作成
    """
    # item_category_name を " - "の前後で大分類と小分類に分ける
    item_categories['main_category'] = item_categories['item_category_name'].map(
        lambda category: category.split(' - ')[0])
    item_categories['sub_category'] = item_categories['item_category_name'].map(make_sub_category)
    train = pd.merge(train, item_categories[['item_category_id', 'main_category', 'sub_category']], how='left')

    # shop_name は都市名の後に " " を挟んで店名が並ぶ形をしているので、 " " の前から地名を抽出する
    shops['city'] = shops['shop_name'].map(lambda shop_name: shop_name.split(' ')[0])
    train = pd.merge(train, shops[['shop_id', 'city']], how='left')

    # 店のオープン月
    for shop_id in list(set(shops['shop_id'])):
        shops.loc[shop_id, 'open_month'] = sales_train[sales_train['shop_id'] == shop_id]['date_block_num'].min()
    train = pd.merge(train, shops[['shop_id', 'open_month']], how='left').fillna({'open_month': 34})

    # 商品のリリース月
    for item_id in list(set(items['item_id'])):
        items.loc[item_id, 'release_month'] = sales_train[sales_train['item_id'] == item_id]['date_block_num'].min()
    train = pd.merge(train, items[['item_id', 'release_month']], how='left').fillna({'release_month': 34})

    # sales_train に item_category_id を加えておく
    sales_train = pd.merge(sales_train, items[['item_id', 'item_category_id']], how='left')

    # 商品ごとの販売価格の中央値
    item_price_median = sales_train[['item_id', 'item_price']].groupby(
        ['item_id'], as_index=False).median().rename(columns={'item_price': 'item_price_median'})
    # カテゴリごとの販売価格の中央値
    item_price_category_mean = sales_train[['item_category_id', 'item_price']].groupby(
        ['item_category_id'], as_index=False).mean().rename(columns={'item_price': 'item_price_median'})
    # 販売価格のデータが存在しない商品は同一カテゴリの商品の平均値を用いる
    train = pd.merge(train, item_price_category_mean, how='left')
    train['item_price_median'].update(pd.merge(train, item_price_median, how='left')['item_price_median'])

    """
    目的変数の作成
    """
    # sales_train から目的変数である月別の販売数を計算(販売実績がない組(月, 店, 商品)には NaN が入るので 0 で埋める)
    item_cnt_month = sales_train[['date_block_num', 'shop_id', 'item_id', 'item_cnt_day']].groupby(
        ['date_block_num', 'shop_id', 'item_id'], as_index=False).sum().rename(columns={'item_cnt_day': 'target'})
    train = pd.merge(train, item_cnt_month, how='left').fillna({'target': 0})

    """
    テストデータの処理
    """
    # test_x に item_category_id を追加
    test_x = pd.merge(test_x, items[['item_id', 'item_category_id']], how='left')

    # test_x に date_block_num を 34(予測しなければならない月を表す) で埋めて追加
    test_x.insert(3, 'date_block_num', 34)

    # test_x に上で作成した特徴量を追加
    test_x = pd.merge(test_x, item_categories[['item_category_id', 'main_category', 'sub_category']], how='left')
    test_x = pd.merge(test_x, shops[['shop_id', 'city']], how='left')
    test_x = pd.merge(test_x, shops[['shop_id', 'open_month']], how='left').fillna({'open_month': 34})
    test_x = pd.merge(test_x, items[['item_id', 'release_month']], how='left').fillna({'release_month': 34})
    test_x = pd.merge(test_x, item_price_category_mean, how='left')
    test_x['item_price_median'].update(pd.merge(test_x, item_price_median, how='left')['item_price_median'])

    """
    label encoding
    """
    # label encoding するカテゴリ変数
    category_value_list = ['main_category', 'sub_category', 'city']

    # train と test_x をlabel encoding
    for category_value in category_value_list:
        # 学習
        le = LabelEncoder()
        le.fit(train[category_value])

        # 変換
        train[category_value] = le.transform(train[category_value])
        test_x[category_value] = le.transform(test_x[category_value])

    """
    型の最適化
    """
    # メモリ節約のため型を最適化しておく
    train = downcast_dtypes(train)
    test_x = downcast_dtypes(test_x)

    """
    ラグ特徴量の作成
    """
    # 組(店, 商品, 月)でソート
    train = train.sort_values(['shop_id', 'item_id', 'date_block_num']).reset_index(drop=True)
    test_x = test_x.sort_values(['shop_id', 'item_id']).reset_index(drop=True)

    # train から test_x に存在する組(店, 商品)に関するデータを抽出する
    test_x_shop_id = set(test_x.shop_id)
    test_x_item_id = set(test_x.item_id)
    train_in_test_x = train.query('shop_id in @test_x_shop_id and item_id in @test_x_item_id')

    # 1~6ヶ月前をラグ特徴量にする
    lag_num_list = list(range(1, 7))

    # 訓練データのラグ特徴量
    for lag in lag_num_list:
        # 追加するラグ特徴量のカラム名
        col_name = "target" + "_" + str(lag)

        # train を lag で指定した回数下にシフトし、 target をリネーム
        # ここでリネームしたカラムがラグ特徴量になる
        train_shifted = train.shift(lag).rename(columns={'target': col_name})

        # シフトした train からラグ特徴量を切り取り、元の train の右側に接着する
        train = pd.concat([train, train_shifted[col_name]], axis=1)

        # シフトした結果、別の商品の値が使われてしまう行には 0 を入れておく
        for index in range(lag):
            train.loc[train['date_block_num'] == index, col_name] = 0

        # 型の最適化
        train = train.astype({col_name: 'uint8'})

    # テストデータのラグ特徴量
    for lag in lag_num_list:
        # 追加するラグ特徴量のカラム名
        col_name = "target_" + str(lag)

        # train_in_test_x から lag で指定した月のデータを抽出し、 target をリネーム
        # ここでリネームしたカラムがラグ特徴量になる
        train_in_test_x_lag = train_in_test_x.query('date_block_num == 34 - @lag').rename(
            columns={'target': col_name})

        # test_x にラグ特徴量を加える
        test_x = pd.merge(test_x, train_in_test_x_lag[['shop_id', 'item_id', col_name]], how='left')

        # テストデータには含まれているが訓練データには含まれていない組(店, 商品)に対してのラグ特徴量は、 0 で埋める
        test_x = test_x.fillna({col_name: 0})

        # 型の最適化
        test_x = test_x.astype({col_name: 'uint8'})

    # ID 順に戻すためソートする
    test_x = test_x.sort_values('ID').reset_index(drop=True)

    """
    特徴量の削除
    """
    # 精度に寄与しない特徴量は削除する
    train = train.drop(['main_category', 'city'], axis=1)
    test_x = test_x.drop(['main_category', 'city'], axis=1)

    """
    処理済みのデータをCSVファイルに保存
    """
    train.to_csv('../preprocessed/train.csv', index=False)
    test_x.to_csv('../preprocessed/test_x.csv', index=False)

    # NaN が 含まれている行がないか表示(デバッグ用)
    print(train.isnull().any())
    print(test_x.isnull().any())

    print("predict_time:{0}".format(time.time() - start) + "[sec]")
