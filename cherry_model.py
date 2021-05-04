import pandas as pd, numpy as np

from sklearn.metrics import r2_score

#fn_cherry = 'http://atmenv.envi.osakafu-u.ac.jp/osakafu-content/uploads/sites/251/2015/10/KyotoFullFlower7.xls'
fn_cherry = '../data/KyotoFullFlower7.xls'

df_cherry = pd.read_excel(
                fn_cherry,
                skiprows=25,
                usecols=['AD', 'Full-flowering date (DOY)'],
                index_col='AD'
        ).dropna().rename(columns={'Full-flowering date (DOY)': 'DOY'})

df_cherry = df_cherry.astype(np.int16)

for deg in range(1, 20):
    model = np.polyfit(
                df_cherry.DOY.index,
                df_cherry.DOY.values, deg)

    predict = np.polyval(model, df_cherry.DOY.index)

    r2 = r2_score(df_cherry.DOY.values, predict)

    print('Degree:', deg, '=> R^2:', r2)
