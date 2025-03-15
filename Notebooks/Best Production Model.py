import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xg
from sklearn.metrics import r2_score

data = pd.read_csv(r'C:\Users\Lenovo\Desktop\AgriCast Predictive Analytics for Crop Production\Data\Scaled\crop_production_scaled.csv').iloc[:10000, :]

x = data.drop(columns=['Production'], axis=1)
y = data['Production']

# Scaling X
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Spliting values for training and testing
xtrain, xtest, ytrain, ytest = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Defining Model
model = xg.XGBRegressor(
    objective='reg:linear',
    n_estimators=300,
    learning_rate=0.01,
    max_depth=None,
    verbose=1,
    n_jobs=-1
)

# Model Training
model.fit(xtrain, ytrain)

# Model Prediction
ypred = model.pridict(xtest)

# Accuract Of Model
print("R2 Score: ", r2_score(ytest, ypred))