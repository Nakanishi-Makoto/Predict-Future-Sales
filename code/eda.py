import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# データフレームのprint時に改行や列の省略がされないように設定
pd.set_option("display.width", 200)
pd.set_option('display.max_columns', 10)

# データ読み込み
item_categories = pd.read_csv('../input/item_categories.csv')
items = pd.read_csv('../input/items.csv')
sales_train = pd.read_csv('../input/sales_train.csv')
shops = pd.read_csv('../input/shops.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/sample_submission.csv')

print("\n")

# item_categories.csv
print("[item_categories]\n", item_categories)
# 欠損値はあるか？
print(item_categories.isnull().any())
# item_category_id の範囲は？
print("item_category_id max:", item_categories['item_category_id'].max())
print("item_category_id min:", item_categories['item_category_id'].min())
# item_category_name に重複はないか？
if set(item_categories['item_category_name'].value_counts()) == {1}:
    print("item_category_name に重複はない")
# item_category_id は一意か？
if set(item_categories['item_category_id'].value_counts()) == {1}:
    print("item_category_id は一意である")

print("\n")

# items.csv
print("[items]\n", items)
# 欠損値はあるか？
print(items.isnull().any())
# item_id の範囲は？
print("item_id max:", items['item_id'].max())
print("item_id min:", items['item_id'].min())
# item_name に重複はないか？
if set(items['item_name'].value_counts()) == {1}:
    print("item_name に重複はない")
# item_id は一意か？
if set(items['item_id'].value_counts()) == {1}:
    print("item_id は一意である")
# item_categories_id は定義されているもののみ使用されているか？
if set(items['item_category_id']) <= set(item_categories['item_category_id']):
    print("item_category_id はすべて定義されている")

print("\n")

# shops.csv
print("[shops]\n", shops)
# 欠損値はあるか？
print(shops.isnull().any())
# shop_id の範囲は？
print("shop_id max:", shops['shop_id'].max())
print("shop_id min:", shops['shop_id'].min())
# shop_name に重複はないか？
if set(shops['shop_name'].value_counts()) == {1}:
    print("shop_name に重複はない")
# shop_id は一意か？
if set(shops['shop_id'].value_counts()) == {1}:
    print("shop_id は一意である")

print("\n")

# sales_train.csv
print("[sales_train]\n", sales_train)
# 欠損値はあるか？
print(sales_train.isnull().any())
# date_block_num max の範囲は？
print("date_block_num max:", sales_train['date_block_num'].max())
print("date_block_num min:", sales_train['date_block_num'].min())
# shop_id の範囲は？
print("shop_id max:", sales_train['shop_id'].max())
print("shop_id min:", sales_train['shop_id'].min())
# item_id の範囲は？
print("item_id max:", sales_train['item_id'].max())
print("item_id min:", sales_train['item_id'].min())
# item_price の範囲は？
print("item_price max:", sales_train['item_price'].max())
print("item_price min:", sales_train['item_price'].min())
# item_cnt_day の範囲は？
print("item_cnt_day max:", sales_train['item_cnt_day'].max())
print("item_cnt_day min:", sales_train['item_cnt_day'].min())
# shop_id は定義されているもののみ使用されているか？
if set(sales_train['shop_id']) <= set(shops['shop_id']):
    print("shop_id はすべて定義されている")
# item_id は定義されているもののみ使用されているか？
if set(sales_train['item_id']) <= set(items['item_id']):
    print("item_id はすべて定義されている")
# 上の結果より、 item_price が -1 のデータポイントが存在することが分かったので、表示してみる
print("item_price < 0:\n", sales_train[sales_train.item_price < 0])

print("\n")

# test.csv
print("[test]\n", test)
# shop_id の範囲は？
print("shop_id max:", test['shop_id'].max())
print("shop_id min:", test['shop_id'].min())
# item_id の範囲は？
print("item_id max:", test['item_id'].max())
print("item_id min:", test['item_id'].min())
# shop_id は定義されているもののみ使用されているか？
if set(test['shop_id']) <= set(shops['shop_id']):
    print("shop_id はすべて定義されている")
# item_id は定義されているもののみ使用されているか？
if set(test['item_id']) <= set(items['item_id']):
    print("item_id はすべて定義されている")

print("\n")

print("[sample_submission]\n", sample_submission)

# 外れ値の有無を確認するため、箱ひげ図で　item_cnt_day　と item_price をプロット
fig, ax = plt.subplots(2, 1, figsize=(10, 4))
# item_price
plt.xlim(-10, 350000)
ax[0].boxplot(sales_train.item_price, labels=['sales_train.item_price'], vert=False)
# item_cnt_day
plt.xlim(-50, 2500)
ax[1].boxplot(sales_train.item_cnt_day, labels=['sales_train.item_cnt_day'], vert=False)
# 箱ひげ図を描画
plt.show()
