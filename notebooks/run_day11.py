import pandas as pd

returns_with_events = pd.read_parquet('../data/returns_with_events.parquet')

# Split chronologically (last year as test set)

# We use the TreeExplainer from SHAP

# Let's compare NH-HMM pure strategy vs XGBoost layered strategy on the TEST set

joblib.dump(classifier, '../src/xgboost_classifier.pkl')
