"""Predict bundle demand and rank historically observed prices by revenue."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
OUT.mkdir(exist_ok=True)
sales = pd.read_csv(ROOT / 'transactionstore.csv')
calendar = pd.read_csv(ROOT / 'dateinfo.csv')
products = pd.read_csv(ROOT / 'selldata.csv')
for frame in (sales, calendar, products):
    frame['CALENDAR_DATE'] = pd.to_datetime(frame['CALENDAR_DATE'], format='mixed')
assert calendar.equals(products), 'The two calendar files disagree.'
keys = ['CALENDAR_DATE', 'SELL_ID', 'SELL_CATEGORY']
bad_dates = sales.loc[sales.duplicated(keys, keep=False), 'CALENDAR_DATE'].unique()
bad_calendar_dates = calendar.loc[calendar.CALENDAR_DATE.duplicated(keep=False), 'CALENDAR_DATE'].unique()
bad_dates = np.union1d(bad_dates, bad_calendar_dates)
excluded = int(sales.CALENDAR_DATE.isin(bad_dates).sum())
sales = sales.loc[~sales.CALENDAR_DATE.isin(bad_dates)].copy()
calendar = calendar.loc[~calendar.CALENDAR_DATE.isin(bad_calendar_dates)].drop_duplicates()
assert not calendar.CALENDAR_DATE.duplicated().any(), 'Conflicting calendar records.'
bundles = products.groupby(['SELL_ID', 'SELL_CATEGORY']).agg(
    COMBINATION=('ITEM_NAME', lambda v: ' + '.join(sorted(set(v)))),
    ITEM_COUNT=('ITEM_ID', 'nunique')).reset_index()
data = sales.merge(calendar, on='CALENDAR_DATE', validate='many_to_one', indicator=True)
assert data['_merge'].eq('both').all(), 'Sales dates missing from calendar.'
data = data.drop(columns='_merge').merge(bundles, on=['SELL_ID','SELL_CATEGORY'], validate='many_to_one')
assert data.COMBINATION.notna().all(), 'Unknown product offering.'
data['DAY_OF_WEEK'] = data.CALENDAR_DATE.dt.dayofweek
data['MONTH'] = data.CALENDAR_DATE.dt.month
data['TIME_INDEX'] = (data.CALENDAR_DATE - data.CALENDAR_DATE.min()).dt.days
data['HOLIDAY'] = data.HOLIDAY.fillna('NONE')
for item in sorted(products.ITEM_NAME.unique()):
    ids = products.loc[products.ITEM_NAME.eq(item), 'SELL_ID']
    data['HAS_' + item] = data.SELL_ID.isin(ids).astype(int)
cat = ['SELL_ID', 'HOLIDAY']
num = ['PRICE','ITEM_COUNT','DAY_OF_WEEK','MONTH','TIME_INDEX','IS_WEEKEND',
       'IS_SCHOOLBREAK','AVERAGE_TEMPERATURE','IS_OUTDOOR'] + [c for c in data if c.startswith('HAS_')]
features = cat + num
def build_model():
    prep = ColumnTransformer([
        ('categories', OneHotEncoder(handle_unknown='ignore'), cat),
        ('numbers', SimpleImputer(strategy='median'), num)])
    return make_pipeline(prep, RandomForestRegressor(
        n_estimators=250, min_samples_leaf=5, random_state=42, n_jobs=-1))
dates = np.sort(data.CALENDAR_DATE.unique())
cutoff = dates[int(len(dates)*0.8)]
train = data.loc[data.CALENDAR_DATE < cutoff]
test = data.loc[data.CALENDAR_DATE >= cutoff]
model = build_model()
model.fit(train[features], train.QUANTITY)
pred = np.maximum(0, model.predict(test[features]))
baseline = test.SELL_ID.map(train.groupby('SELL_ID').QUANTITY.mean()).to_numpy()
evaluation = {'target':'daily quantity', 'objective':'expected daily revenue = price * predicted quantity',
    'train_rows':len(train), 'test_rows':len(test), 'test_start':str(pd.Timestamp(cutoff).date()),
    'test_end':str(test.CALENDAR_DATE.max().date()),
    'quantity_mae':mean_absolute_error(test.QUANTITY,pred),
    'baseline_quantity_mae':mean_absolute_error(test.QUANTITY,baseline),
    'revenue_mae':mean_absolute_error(test.PRICE*test.QUANTITY,test.PRICE*pred),
    'excluded_conflicting_rows':excluded,
    'excluded_dates':[str(pd.Timestamp(d).date()) for d in bad_dates],
    'calendar_duplicate_rows_removed':1349-len(calendar)}
test_output = test[['CALENDAR_DATE','SELL_ID','COMBINATION','PRICE','QUANTITY']].copy()
test_output['PREDICTED_QUANTITY'] = pred
test_output.to_csv(OUT/'held_out_predictions.csv',index=False)
# Refit after evaluation. Rank price scenarios under the latest 30 days of
# calendar conditions. This is a recent-context scenario, not a future forecast.
model = build_model()
model.fit(data[features],data.QUANTITY)
scenario_dates = dates[-30:]
rows = []
for sell_id, group in data.groupby('SELL_ID'):
    context = group.loc[group.CALENDAR_DATE.isin(scenario_dates)].copy()
    for price, count in group.PRICE.value_counts().items():
        if count < 10:
            continue
        scenario = context.copy()
        scenario['PRICE'] = price
        demand = float(np.maximum(0,model.predict(scenario[features])).mean())
        rows.append({'SELL_ID':int(sell_id),'COMBINATION':group.COMBINATION.iloc[0],
                     'ITEM_COUNT':int(group.ITEM_COUNT.iloc[0]),'PRICE':float(price),
                     'PREDICTED_DAILY_QUANTITY':demand,'PREDICTED_DAILY_REVENUE':float(price*demand),
                     'HISTORICAL_DAYS_AT_PRICE':int(count)})
ranking = pd.DataFrame(rows).sort_values('PREDICTED_DAILY_REVENUE',ascending=False)
ranking.to_csv(OUT/'price_scenarios.csv',index=False)
best = ranking.drop_duplicates('SELL_ID')
best.to_csv(OUT/'best_price_per_offering.csv',index=False)
best.loc[best.ITEM_COUNT.ge(2)].to_csv(OUT/'bundle_ranking.csv',index=False)
historical = data.assign(REVENUE=data.PRICE*data.QUANTITY).groupby(['SELL_ID','COMBINATION']).agg(
    DAYS=('QUANTITY','size'),AVERAGE_PRICE=('PRICE','mean'),AVERAGE_DAILY_QUANTITY=('QUANTITY','mean'),
    AVERAGE_DAILY_REVENUE=('REVENUE','mean')).reset_index()
historical.to_csv(OUT/'historical_summary.csv',index=False)
(OUT/'metrics.json').write_text(json.dumps(evaluation,indent=2))
joblib.dump({'model':model,'features':features,'scenario_dates':scenario_dates},OUT/'demand_model.joblib')
print(json.dumps(evaluation,indent=2))
print(best.to_string(index=False))
