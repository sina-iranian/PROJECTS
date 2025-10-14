Car Price Modeling — Data Cleaning, EDA, and Feature Engineering

This notebook prepares a used-car dataset for linear regression by cleaning the data, handling outliers, transforming the target, diagnosing multicollinearity, and encoding categorical variables. It also visualizes key relationships to validate linear model assumptions.

1) Imports & Setup

Load core scientific Python stack:

numpy, pandas for data manipulation

matplotlib.pyplot, seaborn for plots

statsmodels for diagnostics (VIF)

sklearn.linear_model (reserved for later modeling)

Set a simple Seaborn style with sns.set().

2) Load Data

Read the raw CSV into a DataFrame (raw_data), e.g.:

Brand | Price | Body | Mileage | EngineV | Engine Type | Registration | Year | Model


Copy a working dataset and drop the unused Model column:

data = raw_data.drop(['Model'], axis=1)

3) Reindexing Note (when needed)

You only reset the index after dropping rows or filtering/shuffling that creates gaps (e.g., indexes 0, 3, 7).

data.reset_index(drop=True, inplace=True)


Dropping columns does not require reindexing.

4) Initial Summary & Missing Values

data.describe(include='all') to get a high-level summary.

Check NAs with data.isnull().sum().

Handling strategy used here:

Create data_no_mv = data.dropna(axis=0) (simple, keeps only complete cases).

(Commented alternative): keep rows with price and impute EngineV if needed.

5) Outlier Trimming (Robustness Prep)

Trim extreme values to stabilize regression and improve linearity/homoscedasticity:

Price: keep values below the 99th percentile

q = data_no_mv['Price'].quantile(0.99)
data_1 = data_no_mv[data_no_mv['Price'] < q]


Mileage: keep values below its 99th percentile

q = data_1['Mileage'].quantile(0.99)
data_2 = data_1[data_1['Mileage'] < q]


EngineV: cap at a reasonable maximum (domain cap)

data_3 = data_2[data_2['EngineV'] < 6.5]


Year: drop the bottom 1% (very old outliers)

q = data_3['Year'].quantile(0.01)
data_4 = data_3[data_3['Year'] > q]


Reset index after filtering:

data_cleaned = data_4.reset_index(drop=True)


Plotting note: seaborn.distplot is deprecated. Prefer sns.histplot or sns.displot in new code.

6) Target Transformation (Log-Price)

Many price distributions are right-skewed; linear models like symmetry and constant variance.

Transform: data_cleaned['log_price'] = np.log(data_cleaned['Price'])

Drop original Price when modeling with log_price:

data_cleaned = data_cleaned.drop(['Price'], axis=1)

7) Visual Diagnostics (Linearity)

Simple regplots (with log_price) to verify approximate linear trends:

Year vs log_price

EngineV vs log_price

Mileage vs log_price

f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15,3))
sns.regplot(x='Year', y='log_price', data=data_cleaned, color='red', ax=ax1)
sns.regplot(x='EngineV', y='log_price', data=data_cleaned, color='blue', ax=ax2)
sns.regplot(x='Mileage', y='log_price', data=data_cleaned, color='green', ax=ax3)

8) Multicollinearity Check (VIF)

Start with key numeric predictors:

variables = data_cleaned[['Mileage', 'Year', 'EngineV']]


Compute VIF:

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame({
    "VIF": [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])],
    "features": variables.columns
})


Finding: Year shows high VIF (correlated with other features).
Action: drop Year for a low-collinearity baseline:

data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)


Alternative: convert Year → Age = current_year - Year and recheck VIF.

9) Categorical Encoding (Dummies)

One-hot encode all categorical variables:

data_with_dummies = pd.get_dummies(data_cleaned, drop_first=True)


drop_first=True avoids the dummy variable trap by removing one level per category.

Build a modeling table with log_price first, followed by predictors:

cols = ['log_price','Mileage','EngineV','Brand_BMW','Brand_Mercedes-Benz','Brand_Mitsubishi',
        'Brand_Renault','Brand_Toyota','Brand_Volkswagen','Body_hatch','Body_other',
        'Body_sedan','Body_vagon','Body_van','Engine Type_Gas','Engine Type_Other',
        'Engine Type_Petrol','Registration_yes']
data_preprocessed = data_with_dummies[cols]


Convert booleans to numeric:

data_numeric = data_preprocessed.astype(float)

Important VIF Caveats Encountered

Do not include the target (log_price) when computing VIF.
Including it artificially inflates VIF (you briefly saw a very high VIF for log_price).

If you encode all levels of a categorical feature (e.g., both Registration_no and Registration_yes) without dropping a reference level, you will get infinite VIF due to perfect multicollinearity.
Use drop_first=True or manually drop one level per categorical variable.

10) Readiness for Modeling

At this point, you have:

Cleaned, trimmed, and reindexed data (data_cleaned)

A log-transformed target (log_price)

Encoded categorical features (dummy variables with one level dropped)

A numeric, modeling-ready dataset (data_numeric) suitable for:

Statsmodels OLS (for coefficients, p-values, confidence intervals)

scikit-learn LinearRegression (for predictions and pipelines)

Example Next Steps (not run in this notebook)
# X / y split
X = data_numeric.drop(columns=['log_price'])
y = data_numeric['log_price']

# Fit a simple linear model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X, y)

# Coefficients and quick performance
r2 = lr.score(X, y)
print("R^2:", r2)

Key Takeaways

Outlier trimming (quantiles + domain caps) improves stability.

Log transform on price helps linear model assumptions.

VIF is used to diagnose and mitigate multicollinearity (drop Year or use Age).

One-hot encoding with a reference level dropped prevents the dummy trap.

Result: a clean, numeric design matrix ready for linear regression on log_price.
