import pandas as pd
import numpy as np
from math import ceil

# 1. Read the merged CSV file
df = pd.read_csv('ec_merged.csv')

# 2. Prepare lists to hold train/test indices
train_indices = []
test_indices = []

# 3. Fix the random seed for reproducibility
rng = np.random.RandomState(42)

# 4. Iterate over each tag group
for tag, group_df in df.groupby('tag'):
    ids = list(group_df.index)
    n = len(ids)
    if n <= 1:
        # If only one question for this tag, put it in train
        train_indices.extend(ids)
    else:
        # Compute number of test samples (at least 1, ~20% of the group)
        test_count = max(1, int(ceil(0.2 * n)))
        # Randomly sample indices for test from this tag
        chosen = rng.choice(ids, size=test_count, replace=False)
        test_indices.extend(chosen.tolist())
        # The rest go to the training set
        train_indices.extend([i for i in ids if i not in chosen])

# 5. Create train/test DataFrames using the selected indices
train_df = df.loc[train_indices].reset_index(drop=True)
test_df  = df.loc[test_indices].reset_index(drop=True)

# 6. Save the splits to CSV files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
