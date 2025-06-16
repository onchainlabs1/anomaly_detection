import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
df = pd.read_csv('data/creditcard.csv')

# Split into reference and live data
reference_data, live_data = train_test_split(df, test_size=0.2, random_state=42)

# Save the datasets
reference_data.to_csv('data/reference_data.csv', index=False)
live_data.to_csv('data/live_data.csv', index=False)

print("Data split completed successfully!") 