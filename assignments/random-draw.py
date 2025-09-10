import os
import numpy as np
import polars as pl

rng = np.random.default_rng()

setup = 0   # Setup flag
decay = 0.5 # Weight decay rate

roster_file = 'Fall 2025 DATA-5600-001 Course Roster.csv'
roster_save = 'fall-2025_roster.parquet'

# One time setup
if setup == 1:
    draw = (
        pl.read_csv(os.path.join('assignments', roster_file))
        .select('Name')
        .with_columns(drawn = pl.lit(0))
    )

    # Write the roster_save file
    draw.write_parquet(os.path.join('assignments', roster_save))

# Load the roster_save file
draw = pl.read_parquet(os.path.join('assignments', roster_save))

# Compute weights
draw = draw.with_columns(
    weight = (pl.lit(decay) ** pl.col('drawn')).alias('weight')
)

# Turn weights into probabilities
weights = draw['weight'].to_numpy()
probs = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)

# Randomly draw a student
idx = rng.choice(draw.height, p = probs)
picked_row = draw.row(idx, named=True)
picked_name = picked_row['Name']

print(f'The randomly selected student is {picked_name}.')

# Update the drawn count
updated = draw.with_columns(
    pl.when(pl.col('Name') == picked_name)
        .then(pl.col('drawn') + 1)
        .otherwise(pl.col('drawn'))
        .alias('drawn')
).drop('weight')

# Update the roster_save file
updated.write_parquet(os.path.join('assignments', roster_save))

