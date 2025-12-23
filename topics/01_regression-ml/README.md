# Regression and Machine Learning


Welcome to DATA 5600 Introduction to Regression and Machine Learning!
During this class we will:

- Make introductions
- Form groups to work on projects
- Walk through the course syllabus
- Provide some context for what we’ll be studying

## Learn

You can also download the slides as an .html file. Once you’ve previewed
the material and identified any questions, start watching the lecture.

Go the discussions to introduce yourself and starting forming groups of
**two students each** before continuining with the lecture.

Review the syllabus. What questions do you have? What about the course
makes you excited or nervous? Go the discussions to share before
finishing the lecture.

## Case

As a data analyst at Harmons, you are tasked with using product and
customer data to inform a variety of business decisions, including
promotion and pricing strategy, category management, and product line
optimization. Following the soft launch of a new private label peanut
butter, you have been asked to use the resulting data to support a
go/no-go and pricing decision for an expected full launch of the new
product.

The test product supply only permitted a soft launch of one week in a
single designated market area (DMA). All Harmons in the DMA had the
private label in stock and positioned alongside leading national brands.
While promotions varied across locations in the DMA, the stores serve
similar customer segments and are otherwise comparable.

The resulting transaction data has a single observation per customer for
the peanut butter category and includes total spend, the number of units
sold, the brand purchased, the product price, the type of promotion, and
whether the customer was enrolled in Harmons’ loyalty program.

``` python
import os
import numpy as np
import polars as pl
import seaborn.objects as so

# Randomization seed
rng = np.random.default_rng(42)

# Parameter values
n = 4_793
beta_0 = 0.05
beta_jif = 0.20
beta_skippy = 0.15
beta_peterpan = -0.05
beta_harmons = 0.10
beta_coupon = 0.12
beta_ad = 0.08
beta_loyal = 0.15
beta_log_price = -0.90
beta_texture_smooth = 0.10
beta_texture_chunky = -0.10
beta_log_size = 0.10
beta_lp_x_coupon = 0.15
beta_loyal_x_coupon = 0.05
beta_harmons_x_ad = 0.06
beta_jif_x_smooth = 0.05
beta_age = -0.01
beta_log_avg_spend = 0.50
sigma = 0.15

# Customer IDs
cust_ids = np.array([f'C{str(i).zfill(5)}' for i in range(1, n+1)])
rng.shuffle(cust_ids)

# Brands
brands = np.array(['None', 'Jif', 'Skippy', 'PeterPan', 'Harmons'])
brand_probs = np.array([0.40, 0.25, 0.10, 0.05, 0.20])
brand = rng.choice(brands, size=n, p=brand_probs)

# Promotions and loyalty
coupon = rng.binomial(1, 0.30, size=n)
ad = rng.binomial(1, 0.20, size=n)
loyal = rng.binomial(1, 0.40, size=n)

# Texture
texture = rng.choice(np.array(['Smooth', 'Chunky']), size=n, p=[0.65, 0.35])

# Jar size (oz)
size_choices = np.array([12, 16])
size_probs   = np.array([0.30, 0.70])
size = rng.choice(size_choices, size=n, p=size_probs)

# Price by brand and size
price_per_oz = {'None': 0, 'Jif': 0.37, 'Skippy': 0.32, 'PeterPan': 0.30, 'Harmons': 0.20}
price = np.array([
    (price_per_oz[b] * s) for b, s in zip(brand, size)
]).round(2)

# Define different coupon discount rates
discount_rates = np.array([0.10, 0.15, 0.20, 0.25])
discount_probs = np.array([0.40, 0.35, 0.20, 0.05])

# Randomly assign discount rates to each observation
coupon_discount_rate = rng.choice(discount_rates, size=n, p=discount_probs)

# Only apply discount if coupon present and product purchased
price_with_coupon = np.where(
    (coupon == 1) & (brand != 'None'),
    price * (1 - coupon_discount_rate),
    price
).round(2)

# Update the price variable
price = price_with_coupon

# Generate customer ages only for loyal customers
loyal_customer_ages = rng.normal(loc=42, scale=12, size=n)
loyal_customer_ages = np.clip(loyal_customer_ages, 18, 80).round(0).astype(int)

# Average monthly grocery spend (household size proxy)
loyal_avg_spend = rng.lognormal(mean=np.log(180), sigma=0.4, size=n)
loyal_avg_spend = np.clip(loyal_avg_spend, 50, 800).round(0).astype(int)

# Gender - categorical
loyal_gender = rng.choice(['M', 'F', 'Other'], size=n, p=[0.45, 0.52, 0.03])

# Loyalty points earned
loyal_points = rng.exponential(scale=2500, size=n).round(0).astype(int)
loyal_points = np.clip(loyal_points, 0, 15000) # Cap at 15,000 points

# Email preferences
loyal_email_pref = rng.choice(['Yes', 'No'], size=n, p=[0.75, 0.25])

# Clean data for non-buyers
mask_none = (brand == 'None')
texture[mask_none] = 'None'
size[mask_none] = 0
price[mask_none] = 0.0

# Clean data for loyal customers
customer_age = np.where(loyal == 1, loyal_customer_ages, np.nan)
customer_avg_spend = np.where(loyal == 1, loyal_avg_spend, np.nan)
customer_gender = np.where(loyal == 1, loyal_gender, "None")
customer_points = np.where(loyal == 1, loyal_points, np.nan)
customer_email = np.where(loyal == 1, loyal_email_pref, "None")

# Create data frame
raw_df = pl.DataFrame({
    'customer_id': cust_ids,
    'brand': brand,
    'coupon': coupon,
    'ad': ad,
    'loyal': loyal,
    'texture': texture,
    'size': size,
    'price': price,
    'age': customer_age,
    'avg_spend': customer_avg_spend,
    'gender': customer_gender,
    'points': customer_points,
    'email': customer_email,
})

# Dummy code data frame
X_df = raw_df.to_dummies(
    columns = ['brand', 'texture']
).with_columns(
    # Composite promo flag
    ((pl.col('coupon') + pl.col('ad')) > 0).alias('promo'),
    # Log price, size, and avg_spend
    pl.when(pl.col('price') > 0).then(pl.col('price').log()).otherwise(0.0).alias('log_price'),
    pl.when(pl.col('size') > 0).then(pl.col('size').log()).otherwise(0.0).alias('log_size'),
    pl.when(pl.col('avg_spend') > 0).then(pl.col('avg_spend').log()).otherwise(0.0).alias('log_avg_spend'),
).with_columns(
    # Interactions
    (pl.col('log_price') * pl.col('coupon')).alias('logprice_x_coupon'),
    (pl.col('loyal') * pl.col('coupon')).alias('loyal_x_coupon'),
    (pl.col('ad') * pl.col('brand_Harmons')).alias('harmons_x_ad'),
    (pl.col('brand_Jif') * pl.col('texture_Smooth')).alias('jif_x_smooth')
)

# Units and sales
X_df = X_df.with_columns(
    (
        beta_0
        + beta_jif *pl.col('brand_Jif')
        + beta_skippy * pl.col('brand_Skippy')
        + beta_peterpan * pl.col('brand_PeterPan')
        + beta_harmons * pl.col('brand_Harmons')
        + beta_coupon * pl.col('coupon')
        + beta_ad * pl.col('ad')
        + beta_loyal * pl.col('loyal')
        + beta_log_price * pl.col('log_price')
        + beta_log_size * pl.col('log_size')
        + beta_texture_smooth * pl.col('texture_Smooth')
        + beta_texture_chunky * pl.col('texture_Chunky')
        + beta_lp_x_coupon * pl.col('logprice_x_coupon')
        + beta_loyal_x_coupon * pl.col('loyal_x_coupon')
        + beta_harmons_x_ad * pl.col('harmons_x_ad')
        + beta_jif_x_smooth * pl.col('jif_x_smooth')
        + beta_age * pl.col('age').fill_null(0)
        + beta_log_avg_spend * pl.col('log_avg_spend').fill_null(0)
        + rng.normal(0.0, sigma, size=n)
    ).alias('raw_log_units')
).with_columns(
    # Clean log_units and units for non-buyers
    pl.when(pl.col('brand_None') == 1)
        .then(1)
        .otherwise(pl.col('raw_log_units'))
        .alias('log_units')
).with_columns(
    # Exponentiate units and round
    pl.when(pl.col('brand_None') == 1)
        .then(0)
        .otherwise(pl.col('log_units').exp().round(0))
        .alias('units')
).with_columns(
    # Sales
    (pl.col('units') * pl.col('price')).alias('sales'),
)

# Update raw data frame, write
raw_df = raw_df.with_columns(X_df['units', 'sales', 'promo'])

# Write data frames
raw_df.write_parquet(os.path.join('data', 'original_df.parquet'))
X_df.write_parquet(os.path.join('data', 'X_df.parquet'))

# Add missing values at random
missing_loyal_mask = rng.choice([True, False], size=n, p=[0.05, 0.95])
missing_size_mask = rng.choice([True, False], size=n, p=[0.02, 0.98])

# Add common brand name typos
brand_variations = {
    'Jif': ['JIF', 'Jiff'],
    'Skippy': ['SKIPPY', 'Skipy', 'Skipp'],
    'PeterPan': ['Peter Pan'],
    'Harmons': ['Harmon\'s']
}

brand_messy = brand.copy()
for i in range(len(brand_messy)):
    if brand_messy[i] in brand_variations and rng.random() < 0.03:
        brand_messy[i] = rng.choice(brand_variations[brand_messy[i]])

# Add numeric data stored as strings
price_str = price.astype(str)

# Recreate the data frame
raw_df = pl.DataFrame({
    'customer_id': cust_ids,
    'units': X_df['units'],
    'sales': X_df['sales'],
    'brand': brand_messy,
    'promo': X_df['promo'],
    'coupon': coupon,
    'ad': ad,
    'loyal': np.where(missing_loyal_mask, 'None', loyal),
    'texture': texture,
    'size': np.where(missing_size_mask, 'None', size),
    'price': price_str,
    'age': customer_age,
    'avg_spend': customer_avg_spend,
    'gender': customer_gender,
    'points': customer_points,
    'email': customer_email,
})

# Duplicate observations
buyers_mask = raw_df.filter(pl.col('brand') != 'None')
n_duplicates = int(len(buyers_mask) * 0.001)

# Randomly select observations to duplicate and add to the dataset
duplicate_indices = rng.choice(len(buyers_mask), size=n_duplicates, replace=False)
rows_to_duplicate = buyers_mask.slice(duplicate_indices[0], len(duplicate_indices))
raw_df = pl.concat([raw_df, rows_to_duplicate])

# Split into transaction and loyalty data
transaction_df = raw_df.select([
    'customer_id',
    'units',
    'sales', 
    'brand',
    'promo',
    'coupon',
    'ad',
    'loyal',
    'texture',
    'size',
    'price'
])

loyalty_df = (raw_df.select([
    'customer_id', 
    'units',
    'loyal', 
    'age', 
    'avg_spend', 
    'gender', 
    'points', 
    'email'
]).filter(pl.col('loyal') == '1'))

# Write data frames
transaction_df.write_parquet(os.path.join('data', 'soft_launch.parquet'))
loyalty_df.write_parquet(os.path.join('data', 'loyalty.parquet'))
```

## Apply

### Exercise 01

1.  Take the [course
    pre-survey](https://usu.instructure.com/courses/783437/quizzes/1377927)
2.  Start setting up your data stack
3.  Read the case and write how you might go about informing the
    decision (no more than one page)
4.  Include in your response the members of your group
5.  Submit your response as a PDF on Canvas

### Milestone 01

Discuss the project with your group. Identify a business problem that
has data you can acquire. Note that the outcome of this project will be
continuous.
