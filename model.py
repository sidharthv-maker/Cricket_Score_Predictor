import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

data = pd.read_csv("data/cricket_score_prediction_dataset.csv")
y = data.final_score
X = data[['batting_team', 'bowling_team', 'venue', 'overs_completed',  'wickets_fallen', 'current_score', 'current_run_rate', 'last_5_overs_runs', 'batsman1_runs', 'batsman2_runs', 'is_powerplay']]

X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=0)

catcols = ['batting_team', 'bowling_team', 'venue']
numcols = ['overs_completed', 'wickets_fallen', 'current_score', 'last_5_overs_runs','batsman1_runs', 'batsman2_runs', 'is_powerplay']

numpre = SimpleImputer(strategy = 'median')

catpre = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ("num",numpre, numcols),
    ('cat', catpre, catcols)
])

model = RandomForestRegressor(n_estimators=100, random_state=0)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ('model', model)
])

pipeline.fit(X_train,y_train)
pred = pipeline.predict(X_val)

# scores = cross_val_score(pipeline, X, y, cv=5, scoring="neg_mean_absolute_error")
# mae_scores = -scores
# print(mae_scores.mean())

bat_team = input("Please enter the batting team: ")
bowl_team = input("Please enter the bowling team: ")
ven = input("Please enter the venue: ")
over_comp = int(input("How many overs have been completed: "))
wick = int(input("How many wickets have fallen: "))
curr_score = int(input("What is the current score: "))
last_5 = int(input("Runs scored in last 5 overs: "))
bat1 = int(input("Batsman 1 runs: "))
bat2 = int(input("Batsman 2 runs: "))
is_pp = int(input("Is it a powerplay? (1 for yes, 0 for no): "))

user_input = pd.DataFrame([[bat_team, bowl_team, ven, over_comp, wick,curr_score,  last_5, bat1, bat2, is_pp]], columns=['batting_team', 'bowling_team', 'venue','overs_completed', 'wickets_fallen','current_score','last_5_overs_runs', 'batsman1_runs','batsman2_runs', 'is_powerplay'])

ans = pipeline.predict(user_input)
print(f"Predicted score is {ans}")
