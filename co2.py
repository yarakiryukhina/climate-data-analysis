import json, requests, calendar, pandas as pd, numpy as np


def get_json_data(fn, prop=None):
    if fn[:4] == 'http':
        data = requests.get(fn).json()
    else:
        with open(fn) as f:
            data = json.loads(f.read())

    return data[prop] if prop else data


#fn_owid = 'https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv'
fn_owid = '../data/owid-co2-data.csv'

df_owid = pd.read_csv(fn_owid)

# New "other industry" emission calculation grouping by cement, flaring and other

df_owid['other_co2'] = df_owid[['cement_co2', 'flaring_co2', 'other_industry_co2']].sum(axis=1)

world = (df_owid.iso_code == 'OWID_WRL')
others = df_owid.iso_code.isna()


# Global DataFrames: Countries and World

countries_df = df_owid[~world & ~others]
world_df = df_owid[world]


# ———————————————————————————————————————————————————
#   Heatmap by country/year: CO2 share & per capita
# ———————————————————————————————————————————————————

heatmap = pd.pivot(
            countries_df[countries_df.year >= 1990],
            values=['share_global_co2', 'co2_per_capita'],
            index=['country', 'iso_code'],
            columns=['year'])

heatmap_js = {}

for co2_key in ('share_global_co2', 'co2_per_capita'):
    heatmap[co2_key] = round(heatmap[co2_key], 2)

    heatmap_js[co2_key] = []

    for i in heatmap.sort_values(by=[(co2_key, 2019)], ascending=False)[:10].index:
        # i: country name, a: country code
        c = heatmap.loc[i] # Country row
        d = []

        for y in c[co2_key].index:
            d.append({
                'x': str(y),
                'y': c[co2_key][y]
            })

        heatmap_js[co2_key].append({
            'name': i[0], # Country name
            'data': d  # Data: [{x, y, e}]
        })


# ————————————————————————————————————————
#   World CO2 emission by type
# ————————————————————————————————————————

industry = {
    'coal_co2': 'Coal production',
    'oil_co2': 'Oil production',
    'gas_co2': 'Gas production',
    'other_co2': 'Other industrial processes'
}

emission = world_df[['year', 'co2'] + list(industry)].set_index('year').fillna(0)

emission[list(map(lambda x: x+'_pct', industry))] = emission[list(industry)].div(emission.co2, axis=0)

emission_start = 1850

emission_js = {
    'series': [],
    'years': emission.loc[emission_start:].index.to_list()
}

for i in industry:
    d = {
        'name': industry[i],
        'data': round(emission[i + '_pct'].loc[emission_start:] * 100, 1).to_list(),
        'emission': round(emission[i].loc[emission_start:], 1).to_list()
    }

    emission_js['series'].append(d)


# ——————————————————————————————
#   Cherry full-flowering days
# ——————————————————————————————

#fn_cherry = 'http://atmenv.envi.osakafu-u.ac.jp/osakafu-content/uploads/sites/251/2015/10/KyotoFullFlower7.xls'
fn_cherry = '../data/KyotoFullFlower7.xls'

df_cherry = pd.read_excel(
                fn_cherry,
                skiprows=25,
                usecols=['AD', 'Full-flowering date (DOY)'],
                index_col='AD'
        ).dropna().rename(columns={'Full-flowering date (DOY)': 'DOY'})


df_cherry = df_cherry.astype(np.int16)

# Cherry full flowering forecast

cherry_model = np.polyfit(df_cherry.DOY.index, df_cherry.DOY.values, deg=12)

cherry_year = df_cherry.DOY.index[::5].tolist() + \
              list(range(df_cherry.DOY.index[-1] + 1, df_cherry.DOY.index[-1] + 1 + 30))

cherry_predict = pd.Series(np.polyval(cherry_model, cherry_year), index=cherry_year)

cherry2_js = {
    'min': int(df_cherry.DOY.min()),
    'max': int(df_cherry.DOY.max()),
    'series': [{
        'name': 'Historical data',
        'type': 'scatter',
        'data': [{'x': x, 'y': y} for x, y in df_cherry.DOY.items()]
    }, {
        'name': 'Trend',
        'type': 'line',
        'data': [{'x': int(x), 'y': float(y)} for x, y in zip(cherry_predict.index, cherry_predict.round(2))]
    }],
    'markers': [4, 0]
}

# —————————————————————————————————————————
#   Global warming API: CO2 concentration
# —————————————————————————————————————————

#fn_conc = 'https://global-warming.org/api/co2-api'
fn_conc = '../data/co2-api.json'

df_conc = pd.DataFrame(get_json_data(fn_conc, prop='co2')).astype({'year': int, 'month': int, 'day': int, 'cycle': float})

df_conc_mm = df_conc[df_conc.day == 1].copy()

del df_conc

df_conc_mm['diff'] = df_conc_mm['cycle'].diff()
df_conc_mm['month_diff'] = (df_conc_mm['month'] - 1).map(lambda x: x if x > 0 else 12)

df_conc_mm.dropna(inplace=True)

df_conc_diff = df_conc_mm.groupby('month_diff').agg(min_diff=('diff', 'min'),
                                                    max_diff=('diff', 'max'))
conc_diff_js = {
    'data_1': round(df_conc_diff.min_diff, 2).to_list(),
    'data_2': round(df_conc_diff.max_diff, 2).to_list(),
    'category': df_conc_diff.index.to_series().apply(lambda x: calendar.month_abbr[x]).to_list()
}

del df_conc_diff
del df_conc_mm

# ———————————————————————————————————————————
#   Global warming API vs OWID CO2 emission
# ———————————————————————————————————————————

warming_data = []

# 1. Arctic ice

# fn_ice = 'https://global-warming.org/api/arctic-api'
fn_ice = '../data/arctic-api.json'

warming_data.append(pd.DataFrame(get_json_data(fn_ice, prop='result')).astype({'year': int, 'area': float}).set_index('year')['area'])

# 2. Temperature

#fn_temp = 'https://global-warming.org/api/temperature-api'
fn_temp = '../data/temperature-api.json'

temp_df = pd.DataFrame(get_json_data(fn_temp, prop='result'),
                    columns=['time', 'land']).rename(columns={'land': 'temp'})

temp_df[['year', 'month']] = temp_df.time.str.split('.', expand=True)

warming_data.append(temp_df[['year', 'temp']].astype({'year': int, 'temp':float}).groupby('year').last()['temp'])

del temp_df

# 3. Methane & Nitrous oxide concentrations

fn_mn2o = {
    'methane': '../data/methane-api.json',
    #'methane': 'https://global-warming.org/api/methane-api',
    'nitrous': '../data/nitrous-oxide-api.json'
    #'nitrous': 'https://global-warming.org/api/nitrous-oxide-api'
}

for col, fn in fn_mn2o.items():
    mn2o_df = pd.DataFrame(get_json_data(fn, prop=col), columns=['date', 'average']).rename(columns={'average': col})

    mn2o_df[['year', 'month']] = mn2o_df.date.str.split('.', expand=True)

    warming_data.append(mn2o_df[['year', col]].astype({'year': int, col:float}).groupby('year').last()[col])

del mn2o_df

# Merging OWID CO2 emission data with Warming API data

df_warming = round(emission[['co2']]).astype(int)

for w in warming_data:
    df_warming = df_warming.merge(w, left_index=True, right_index=True, how='left')

del warming_data

# Greenhouse gases  concentration change for past 25 years

gh_year = df_warming.index[-1] - 25

greenhouse_js = {
    'methane': {
        'data': round(df_warming.loc[gh_year:].methane /
                      df_warming.loc[gh_year:].methane.min() * 100, 1).to_list(),
        'level': round(df_warming.loc[gh_year:].methane, 2).to_list(),
        'min': float(round(df_warming.loc[gh_year:].methane.min(), 2))
    },
    'nitrous': {
        'data': round(df_warming.loc[gh_year:].nitrous /
                      df_warming.loc[gh_year:].nitrous.min() * 100, 1).to_list(),
        'level': round(df_warming.loc[gh_year:].nitrous, 2).to_list(),
        'min': float(round(df_warming.loc[gh_year:].nitrous.min(), 2))
    },
    'category': df_warming.loc[gh_year:].index.astype(str).to_list()
}

# Arctic ice area, CO2 emission, temperature

ai_year = 1991

arcticice_js = {
    'ice': {
        'data': round(df_warming.loc[ai_year:].area /
                      df_warming.loc[ai_year:].area.max() * 100, 1).to_list(),
        'level': round(df_warming.loc[ai_year:].area, 2).to_list(),
        'max': float(round(df_warming.loc[ai_year:].area.max(), 2))
    },
    'co2': {
        'data': round(df_warming.loc[ai_year:].co2 /
                     df_warming.loc[ai_year:].co2.min() * 100, 1).to_list(),
        'level': round(df_warming.loc[ai_year:].co2, 2).to_list(),
        'min': float(round(df_warming.loc[ai_year:].co2.min(), 2))
    },
    'temp': {
        'data': round(df_warming.loc[ai_year:].temp, 2).to_list(),
    },
    'category': df_warming.loc[ai_year:].index.astype(str).to_list()
}


# ——————————————————————————————————————————————————————
#   CO2 emission per one consumed exajoule. BP dataset
# ——————————————————————————————————————————————————————

#fn_bp = 'https://www.bp.com/content/dam/bp/business-sites/en/global/corporate/xlsx/energy-economics/statistical-review/bp-stats-review-2020-consolidated-dataset-panel-format.csv'
fn_bp = '../data/bp-stats-review-2020-panel.csv'

df_bp = pd.read_csv(fn_bp, usecols=['Country', 'Year', 'ISO3166_alpha3', 'co2_mtco2', 'renewables_ej', 'coalcons_ej', 'gascons_ej', 'oilcons_ej', 'nuclear_ej', 'hydro_ej'])

df_bp['cons_ej'] = df_bp.iloc[:, 4:].sum(axis=1)
df_bp['co2_per_ej'] = df_bp.co2_mtco2 / df_bp.cons_ej

df_bp.update(
    pd.concat([
        df_bp[df_bp.Country == 'USSR'][['co2_per_ej']].dropna(),
        df_bp[df_bp.Country == 'Russian Federation'][['co2_per_ej']].dropna()
    ]).reset_index(drop=True).set_index(df_bp[df_bp.Country == 'Russian Federation'].index)
)

co2_per_ej = {
    'series': [{
        'name': 'World total',
        'data': round(df_bp[df_bp.Country == 'Total World'].co2_per_ej, 1).to_list()
    }],
    'categories': df_bp[df_bp.Country == 'Total World'].Year.to_list(),
    'width': [4],
    'dash': [0]
}

for c, a3 in heatmap.sort_values(by=[('share_global_co2', 2019)], ascending=False)[:10].index:
    co2_per_ej['series'].append({
        'name': c,
        'data': round(df_bp[df_bp.ISO3166_alpha3 == a3].co2_per_ej, 1).tolist()
    })

    co2_per_ej['width'].append(1)
    #co2_per_ej['dash'].append(5),
    co2_per_ej['dash'].append(int(round(100 / df_bp[df_bp.ISO3166_alpha3 == a3].co2_per_ej.iloc[-1], 0) < 2) * 2),


# ——————————————————————————————————————————
#   Energy consumption by type. BP dataset
# ——————————————————————————————————————————

df_energy = df_bp[df_bp.Country == 'Total World'].set_index('Year').loc[df_bp.Year.unique()[-1] - 40:]

energy_series = {
    'renewables_ej': 'Renewables',
    'hydro_ej': 'Hydro',
    'coalcons_ej': 'Coal',
    'gascons_ej': 'Gas',
    'oilcons_ej': 'Oil',
    'nuclear_ej': 'Nuclear'
}

energy_js = {
    'categories': df_energy.index[::-1].tolist(),
    'series': []
}

for e in energy_series:
    energy_js['series'].append({
        'name': energy_series[e],
        'data': (df_energy.loc[::-1][e] / df_energy.loc[::-1].cons_ej * 100).round(3).astype(float).to_list()
    })

# ——————————————————————
#   Output
# ——————————————————————

main_js = {
    'heatmap':    heatmap_js,
    'co2_per_ej': co2_per_ej,
    'energy':     energy_js,
    'emission':   emission_js,
    'cherry2':    cherry2_js,
    'conc_diff':  conc_diff_js,
    'greenhouse': greenhouse_js,
    'arcticice':  arcticice_js
}

if __name__ == '__main__':
    with open('res.json', 'w') as f:
        f.write(json.dumps(main_js, indent=None, ensure_ascii=False).replace('NaN', 'null'))
else:
    print(main_js)
