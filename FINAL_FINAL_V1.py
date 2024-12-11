import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from xgboost import plot_importance, XGBRegressor
from skrub import TableVectorizer
import matplotlib.pyplot as plt
import holidays

# ---------------- DEFINE THE FUNCTIONS --------------------------------

def weather_cleaning(weather):
    weather = weather.drop_duplicates()
    weather = weather[['date', 'ff', 't', 'ssfrai', 'etat_sol', 'ht_neige', 'rr1', 'rr3', 'rr6']]
    weather = weather.interpolate(method='linear', limit_direction='both')

    return weather


def extract_date_features(data, date_column='date'):
    data['hour'] = data[date_column].dt.hour
    data['weekday'] = data[date_column].dt.dayofweek
    data['month'] = data[date_column].dt.month
    data['year'] = data[date_column].dt.year
    data['weekend_day'] = data['weekday'].apply(lambda x: 1 if x in [5, 6] else 0)
    data['season'] = data['month'].apply(
        lambda x: 'spring' if x in [3, 4, 5]
        else 'winter' if x in [12, 1, 2]
        else 'summer' if x in [6, 7, 8]
        else 'autumn'
    )
    france_holidays = holidays.France(years=data['year'].unique())
    data['holidays'] = data[date_column].apply(lambda d: 1 if d in france_holidays else 0)
    return data

def add_rush_hour(data):
    data['is_rush_hour'] = data['hour'].apply(lambda x: 1 if (7 <= x <= 9 or 17 <= x <= 19) else 0)
    return data

def merge_external_data(data, weather):
    data['date'] = pd.to_datetime(data['date']).astype('datetime64[ns]')
    weather['date'] = pd.to_datetime(weather['date']).astype('datetime64[ns]')
    data["orig_index"] = np.arange(data.shape[0])
    merged_df = pd.merge_asof(
        data.sort_values("date"),
        weather.sort_values("date"),
        on="date"
    )
    merged_df = merged_df.sort_values("orig_index")
    del merged_df["orig_index"]
    return merged_df

def strikes (data):
    greves = pd.read_csv("mouvements-sociaux-depuis-2002.csv", sep=';')
    greves = greves[(greves['Date'] > '2020-09-01')]
    # only keep rows with date before 19 october 2021
    greves = greves[(greves['Date'] < '2021-10-19')]
    data['strike'] = data['date'].isin(greves['Date']).astype(int)
    return data

#  ----------------- LOAD AND DEFINE THE FUNCTIONS --------------------------------


data = pd.read_parquet('data/train.parquet')
weather = pd.read_csv('external_data/external_data.csv', parse_dates=["date"])


weather = weather_cleaning(weather)

data['date'] = pd.to_datetime(data['date'])
weather['date'] = pd.to_datetime(weather['date'])

data = extract_date_features(data)
data = strikes(data)
data = add_rush_hour(data)


# ----------------- MERGE DATA --------------------------------

data['date'] = pd.to_datetime(data['date']).astype('datetime64[ns]')
weather['date'] = pd.to_datetime(weather['date']).astype('datetime64[ns]')

merged_data = merge_external_data(data, weather)

# ----------------- DROP UNNECESSARY COLUMNS --------------------------------

y = merged_data['log_bike_count']
X = merged_data.drop(columns=['bike_count', 'log_bike_count', 'counter_id',
                            'site_id', 'coordinates', 'counter_technical_id',
                            'site_name', 'date'])


# ----------------- SPLIT DATA --------------------------------

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=42)




# ----------------- PIPELINE -------------------------------------------------

# Model
cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
date_variables = ['year', 'month', 'weekend_day', 'weekday', 'hour', 'counter_installation_date']

preprocess = ColumnTransformer(
    transformers=[
        ('categorical', cat_encoder, ['counter_name', 'season']),
        ('date', OneHotEncoder(handle_unknown="ignore"), date_variables)
    ],
    remainder='passthrough'
)

xgb = XGBRegressor(random_state=42)

pipe = Pipeline(steps=[
                ('preprocess', preprocess),
                ('regressor', xgb)
])


param_grid = {
    'regressor__max_depth': [5, 8, 15],
    'regressor__n_estimators': [700, 750, 800],
    'regressor__learning_rate': [0.1, 0.01]
}

grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)

grid_search_results = grid_search.fit(X_train, y_train)
print("The best parameters are ",grid_search.best_params_)

xgb = grid_search.best_estimator_
xgb.fit(X_train, y_train)


y_valid_pred = xgb.predict(X_valid)
rmse = MSE(y_valid, y_valid_pred)
print(f"The RMSE is: {np.sqrt(rmse)}")


# ----------------- FINAL TESTING --------------------------------

X_test = pd.read_parquet('data/final_test.parquet')

X_test['date'] = pd.to_datetime(X_test['date'])

merged_test_data = merge_external_data(X_test, weather)

merged_test_data = extract_date_features(merged_test_data)

merged_test_data = strikes(merged_test_data)
merged_test_data = add_rush_hour(merged_test_data)

y_pred = xgb.predict(merged_test_data)
y_pred = np.where(y_pred < 0, 0, y_pred)

sol = {
    'Id': list(range(len(y_pred))),
    'log_bike_count': y_pred.flatten()
}

submission = pd.DataFrame(sol)
submission.set_index("Id", inplace=True)
submission.to_csv('submission.csv')