import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    for colname in X.select_dtypes(["category", "object"]):
        X[colname], _ = X[colname].factorize()
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_log_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

# Prepare data
df = pd.read_csv("../input/fe-course-data/ames.csv")
X = df.copy()
y = X.pop("SalePrice")

# Create math tranforms
X_1 = pd.DataFrame()  # dataframe to hold new features
X_1["LivLotRatio"] = df.GrLivArea / df.LotArea
X_1["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
X_1["TotalOutsideSF"] = df.WoodDeckSF + df.OpenPorchSF + df.EnclosedPorch + df.Threeseasonporch + df.ScreenPorch
X_1[["LivLotRatio","Spaciousness"]].head()

# Create interation features between BldgType and GrLivArea
# One-hot encode BldgType
X_2 = pd.get_dummies(df.BldgType, prefix="Bldg")
X_2 = X_2.mul(df.GrLivArea, axis=0)
# Get Multiplication of dataframe and other, element-wise.

# Count how many of the following are > 0
X_3 = pd.DataFrame()
X_3["PorchTypes"] = df[[
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "Threeseasonporch",
    "ScreenPorch",
]].gt(0.0).sum(axis=1)

# MSSubClass describes the type of a dwelling
df.MSSubClass.unique()

# Create a feature containing only the first word of each type
X_4 = pd.DataFrame()
X_4["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]

# Group transform
X_5 = pd.DataFrame()
X_5["MedNhbdArea"] = (df.groupby("Neighborhood")['GrLivArea'].transform('median'))

# Join features
X_new = X.join([X_1, X_2, X_3, X_4, X_5])
score_dataset(X_new, y)

##########################################################################

# Data visualization can suggest transformations,
# often a "reshaping" of a feature through powers or logarithms.
# The distribution of WindSpeed in US Accidents is highly skewed,
# for instance. In this case the logarithm is effective at normalizing it:
# If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log
accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)
# Plot a comparison
fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, shade=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, shade=True, ax=axs[1]);

# Counts
# Binary features (1 for Present, 0 for Absent) or boolean (True or False)
# In Python, booleans can be added up just as if they were integers.
roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
    "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
    "TrafficCalming", "TrafficSignal"]
accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)
accidents[roadway_features + ["RoadwayFeatures"]].head(10)

# Break down features
customer[["Type", "Level"]] = (  # Create two new features
    customer["Policy"]           # from the Policy feature
    .str                         # through the string accessor
    .split(" ", expand=True)     # by splitting on " "
                                 # and expanding the result into separate columns
)
customer[["Policy", "Type", "Level"]].head(10)

# Join features
autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
autos[["make", "body_style", "make_and_style"]].head()

# Group Transforms
# methods: max, min, mean, median, var, std, count
customer["AverageIncome"] = (
    customer.groupby("State")  # for each state
    ["Income"]                 # select the income
    .transform("mean")         # and compute its mean
)
customer[["State", "Income", "AverageIncome"]].head(10)

customer["StateFreq"] = (
    customer.groupby("State")
    ["State"]
    .transform("count")
    / customer.State.count()
)
customer[["State", "StateFreq"]].head(10)


# Create a "frequency encoding" for a categorical feature
# If you're using training and validation splits, to preserve their independence,
# it's best to create a grouped feature using only the training set
# and then join it to the validation set. We can use the validation set's merge method
# after creating a unique set of values with drop_duplicates on the training set:

# Create splits
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

# Create the average claim amount by coverage type, on the training set
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left",
)

df_valid[["Coverage", "AverageClaim"]].head(10)

# Create splits
df_train = customer.sample(frac=0.5)
df_valid = customer.drop(df_train.index)

# Create the average claim amount by coverage type, on the training set
df_train["AverageClaim"] = df_train.groupby("Coverage")["ClaimAmount"].transform("mean")

# Merge the values into the validation set
df_valid = df_valid.merge(
    df_train[["Coverage", "AverageClaim"]].drop_duplicates(),
    on="Coverage",
    how="left",
)

df_valid[["Coverage", "AverageClaim"]].head(10)
