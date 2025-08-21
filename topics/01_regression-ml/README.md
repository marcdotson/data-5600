# 01 Regression and Machine Learning


## Case Overview

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

# # Parameter values
n = 4_793
beta_0 = 90 / 100
beta_jif = 30 / 100
beta_skippy = 25 / 100
beta_peterpan = -15 / 100
beta_harmons = 20 / 100
beta_coupon = 25 / 100
beta_ad = 18 / 100
beta_loyal = 22 / 100
beta_log_price = -90 / 100
beta_texture_smooth = 10 / 100
beta_texture_chunky = -10 / 100
beta_log_size = 10 / 100
beta_lp_x_coupon = 15 / 100
beta_loyal_x_coupon = 5 / 100
beta_harmons_x_ad = 6 / 100
beta_jif_x_smooth = 5 / 100
sigma = 15 / 100

# Customer IDs
cust_ids = np.array([f'C{str(i).zfill(5)}' for i in range(1, n+1)])
rng.shuffle(cust_ids)

# Brands
brands = np.array(['None', 'Jif', 'Skippy', 'PeterPan', 'Harmons'])
brand_probs = np.array([0.40, 0.25, 0.10, 0.05, 0.20])
brand = rng.choice(brands, size=n, p=brand_probs)

(pl.DataFrame(brand, schema=['brand'])
    .group_by(pl.col(['brand']))
    .agg(n = pl.len())
)

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

# Clean data for non-buyers
mask_none = (brand == 'None')
texture[mask_none] = 'None'
size[mask_none] = 0
price[mask_none] = 0.0

# Create data frame
raw_df = pl.DataFrame(
    {
        'customer_id': cust_ids,
        'brand': brand,
        'coupon': coupon,
        'ad': ad,
        'loyal': loyal,
        'texture': texture,
        'size': size,
        'price': price,
    }
)

# (raw_df
#     .group_by(pl.col(['brand', 'texture']))
#     .agg(n = pl.len())
# )

# (raw_df
#     .group_by(pl.col(['brand', 'size']))
#     .agg(n = pl.len())
# )

# (raw_df
#     .group_by(pl.col(['size', 'texture']))
#     .agg(n = pl.len())
# )

# (raw_df
#     .group_by(pl.col(['brand', 'size']))
#     .agg(
#         avg_price = pl.col('price').mean()
#     )
# )

# Dummy code data frame
X_df = raw_df.to_dummies(
    columns = ['brand', 'texture']
).with_columns(
    # Composite promo flag
    ((pl.col('coupon') + pl.col('ad')) > 0).alias('promo'),
    # Log price and size
    pl.when(pl.col('price') > 0).then(pl.col('price').log()).otherwise(0.0).alias('log_price'),
    pl.when(pl.col('size') > 0).then(pl.col('size').log()).otherwise(0.0).alias('log_size'),
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

# (so.Plot(X_df, x = 'units')
#     .add(so.Bar(), so.Hist())
# )

# (so.Plot(X_df, x = 'log_units')
#     .add(so.Bar(), so.Hist())
# )

# (so.Plot(X_df, x = 'sales')
#     .add(so.Bars(), so.Hist())
# )

# (so.Plot(X_df, x = 'price', y = 'units')
#     .add(so.Dot(alpha = 0.5), so.Jitter(x = 0.5, y = 0.5))
#     .add(so.Line(), so.PolyFit(order=1))
#     .add(so.Band(), so.PolyFit(order=1))
# )

# test_df = X_df.filter(pl.col('units') > 0)

# (so.Plot(test_df, x = 'price', y = 'units')
#     .add(so.Dot(alpha = 0.5), so.Jitter())
#     .add(so.Line(), so.PolyFit(order=1))
#     .add(so.Band(), so.PolyFit(order=1))
# )

# Update data frame
raw_df = raw_df.with_columns(X_df['units', 'sales', 'promo'])

# Add complications
# - Missing values
# - Missing data

# Write data frames
raw_df.write_parquet(os.path.join('data', 'raw_df.parquet'))
X_df.write_parquet(os.path.join('data', 'X_df.parquet'))
```
