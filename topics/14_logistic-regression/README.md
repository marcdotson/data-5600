# Logistic Regression


## Data Dictionary

Sales leads data for B2B electronic sales. Each of the rows represents
one lead along with its resulting Stage (i.e., Qualified or
Disqualified) along with various lead features.

- **Stage**: Progress through the sales development pipeline
- **Industry**: The type of product or service the business provides
- **Employees**: The size of company based on the number of employees
- **TimeZone**: The Lead’s time zone
- **LeadSource**: The source of the data for the Lead
- **days_elapsed**: Number of days elapsed since Lead creation
- **created_quarter**: Quarter of the Lead creation
- **contact_quarter**: Quarter when the Lead was first contacted
- **latest_quarter**: Quarter when the Lead was last contacted
- **EmployeeId**: Unique ID for the sales rep assigned to the Lead
- **ActivityTypeEmail**: Indicator of email contact with the Lead
- **ActivityTypePhone Call**: Indicator of call contact with the Lead
- **ActivityTypeEmail Response**: Indicator of email response with the
  Lead
- **ActivityTypeMeeting**: Indicator of meeting with the Lead
- **ActivityTypeLead Handraise**: Indicator of Lead requesting
  information
- **ActivityTypeWeb Schedule**: Indicator of Lead scheduling an
  appointment
- **Amount**: Estimate of how much the Lead is worth if closed

``` python
import os
import polars as pl
import polars.selectors as cs
import seaborn.objects as so
import statsmodels.formula.api as smf
import bambi as bmb
import arviz as az

# Import data
leads = (pl.read_parquet(os.path.join('data', 'original_leads.parquet'))
    .select(['Stage', 'Industry', 'Employees', 'TimeZone', 'LeadSource',
        'days_elapsed', 'created_quarter', 'contact_quarter', 'latest_quarter',
        'EmployeeId', cs.starts_with('ActivityType'), 'Amount'])
    .select(pl.exclude(['ActivityTypeAbandon']))
)

# Write data frames
leads.write_parquet(os.path.join('data', 'leads.parquet'))
```
