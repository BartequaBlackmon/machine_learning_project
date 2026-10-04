# Product combination and price model

## Result
Among observed bundles, BURGER + COFFEE + COKE ranks first at price 10.81, with predicted daily quantity 50.65 and daily revenue 547.48. Currency is not specified in the files. The burger alone ranks higher in revenue, but is not a bundle.

## Run
Extract this archive, open a terminal in its folder, then run:

    python -m pip install -r requirements.txt
    python train_model.py

The script includes input data and produces a trained model and CSV results in results/.

## Method
Predict QUANTITY with RandomForestRegressor from price, offering ID, component flags, calendar and weather. Multiply predicted quantity by price. Search only historical prices observed on at least 10 valid days, and rank average scenario revenue under the latest 30 historical days (2015-08-12 to 2015-09-10). Offerings are modeled separately; predictions should not be summed into a menu revenue forecast.

SELL_ID is treated as an offering identifier, not a customer basket. Product definitions are grouped before joining, preventing duplication of bundle revenue. Join keys are date for calendar and SELL_ID plus SELL_CATEGORY for offerings. The oddly named selldata (version 1).xlsb.csv is another calendar file; its normalized contents match dateinfo.csv and are checked rather than appended. Both calendars and sales have conflicting entries for 2013-03-01; all 16 sales rows and both calendar rows for that date are excluded. No other data was fabricated.

## Validation
Chronological split: oldest 80% of distinct dates for training, latest 20% for testing. 4308 training rows and 1080 test rows. Held-out daily quantity mean absolute error: 3.16 units; historical per-offering mean baseline: 7.77 units. Revenue MAE: 39.86 currency units per offering per day. Actual calendar/weather conditions are used in testing, so this is conditional demand prediction rather than a fully independent weather forecast. After evaluation, refit on all clean data for scenario ranking. Evaluation measures quantity prediction, not optimal-price accuracy.

## Limitations
Assumes PRICE is the selling price per bundle/unit and QUANTITY is units sold per offering per day. Confirm these definitions before business use. Historical dates are 2012-2015, not current-market demand. No costs, margins, competitor prices, stock availability, promotions or customer-level data. This optimizes modeled revenue, not profit or price alone. Price changes coincide with time and other factors; counterfactual estimates may be confounded. These are exploratory prices to test, not proven optimal prices. Only existing offerings with sales histories are ranked; unseen product combinations lack training evidence. Controlled price experiments and current data are needed for deployment.

## Outputs
- bundle_ranking.csv: best observed-price scenario for each bundle.
- best_price_per_offering.csv: includes burger alone.
- price_scenarios.csv: every candidate scenario.
- historical_summary.csv: actual historical results.
- held_out_predictions.csv and metrics.json: evaluation evidence.
- demand_model.joblib: trained pipeline and feature names. Load only a trusted model file using matching package versions.
