import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_palette("winter", 8)

train = pd.read_csv('../preprocessed/train.csv')

# x軸を月、y軸を全店の販売個数としてプロット
plt_df = train.groupby(['date_block_num'], as_index=False).sum()
plt.figure(figsize=(20, 10))
plt.title('target')
sns.lineplot(x='date_block_num', y='target', data=plt_df)
plt.savefig('../visual/target.png')

# x軸を月、y軸を全店の販売個数として商品の大分類ごとにプロット
plt_df = train.groupby(['date_block_num', 'main_category'], as_index=False).sum()
plt.figure(figsize=(20, 10))
plt.title('target by main_category')
sns.lineplot(x='date_block_num', y='target', data=plt_df, hue='main_category')
plt.savefig('../visual/target by main_category.png')

# x軸を月、y軸を全店の販売個数として地域ごとにプロット
plt_df = train.groupby(['date_block_num', 'city'], as_index=False).sum()
plt.figure(figsize=(20, 10))
plt.title('target by city')
sns.lineplot(x='date_block_num', y='target', data=plt_df, hue='city')
plt.savefig('../visual/target by city.png')
