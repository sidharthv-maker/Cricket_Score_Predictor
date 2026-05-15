# Cricket_Score_Predictor
A simple Machine Learning project that predicts the final score of a cricket innings based on the current match situation.

## Project Overview

This project uses a `RandomForestRegressor` model to predict the final innings score using features such as batting team, bowling team, venue, overs completed, wickets fallen, current score, recent runs, and batsman scores.

The project also uses preprocessing techniques like imputation and one-hot encoding through a scikit-learn pipeline.

## Dataset

The dataset contains cricket innings data with columns such as:

- `batting_team`
- `bowling_team`
- `venue`
- `overs_completed`
- `wickets_fallen`
- `current_score`
- `current_run_rate`
- `last_5_overs_runs`
- `batsman1_runs`
- `batsman2_runs`
- `is_powerplay`
- `final_score`

The target variable is:

```text
final_score
```
## Example Input
```Please enter the batting team: CSK
Please enter the bowling team: MI
Please enter the venue: Wankhede
How many overs have been completed: 10
How many wickets have fallen: 2
What is the current score: 89
Runs scored in last 5 overs: 45
Batsman 1 runs: 45
Batsman 2 runs: 32
Is it a powerplay? (1 for yes, 0 for no): 0
```
## Example Output
```Predicted score is [178.5]```
## Future Improvements
Add more real cricket match data
Include player-level features
Add toss, pitch, and weather conditions
