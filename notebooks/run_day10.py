import pandas as pd

verified_df = pd.read_csv('../data/disagreement_table_verified.csv')

returns_with_events = pd.read_parquet('../data/returns_with_events.parquet')

# Strategy Rule: 

# Let's plot the cumulative returns overall
