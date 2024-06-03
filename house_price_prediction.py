import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


df = pd.read_csv('data.csv')

X = df.drop(['date','street','city','statezip','country'],axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
continous_features_used = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "view", "condition", "sqft_above", "sqft_basement", "yr_built", "yr_renovated"]
ct = ColumnTransformer(
                        [('scaler', StandardScaler(), continous_features_used)]
                       ,remainder="passthrough")

model = Pipeline([('columnTransform', ct), ('model', ElasticNet(alpha=0.001, l1_ratio=0.0,max_iter=1000))])
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:",r2)