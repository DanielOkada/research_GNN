import pandas as pd

# credits.csvを読み込む
credits = pd.read_csv('data/credits.csv')

# castデータのみを取り出す
cast = credits['cast']

# キャスト情報を正規化する
cast = pd.DataFrame(cast.values.tolist(), index=credits.index)

print(cast)
# 必要な列のみを取り出す
cast = cast[[0,1,2,3]]
cast.columns = ['cast_id', 'character', 'credit_id', 'id']