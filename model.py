import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from imblearn.under_sampling import TomekLinks

import pickle

     
df = pd.read_csv("input_data.csv")


columns_to_map = ["location", "city", "stacked_rest_type", "stacked_cuisines", "category"]
ordinal_cols = ['online_order', 'book_table']  
ordinal_categories = {
    'online_order': ['No', 'Yes'], 
    'book_table': ['No', 'Yes']   
}
ordinal_encoder = OrdinalEncoder(categories=[ordinal_categories[col] for col in ordinal_cols])
# Define a function to generate mappings for each column
def generate_mapping(column):
    return {j: i for i, j in enumerate(column.unique())}

# Custom function to apply the mappings
def apply_mapping(data, mapping):
    for col, map_dict in mapping.items():
        data[col] = data[col].map(map_dict)
    return data

# Generate mappings for each column
mappings = {col: generate_mapping(df[col]) for col in columns_to_map}
 
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric_imputer', SimpleImputer(strategy='mean'), ["approx_cost"]),
        ('map_columns', FunctionTransformer(apply_mapping, kw_args={'mapping': mappings}), columns_to_map),
        ('ordinal', ordinal_encoder, ordinal_cols),
    ],
    remainder='passthrough'
)


#split data 
X = df.drop("status", axis=1)
y = df["status"]

 
#LogisticRegression 
base_models = [
    ('rf', RandomForestClassifier(max_depth=7, max_features='sqrt', min_samples_leaf=10,
                                  min_samples_split=10, n_estimators=50)),
    ('lr', DecisionTreeClassifier())
]

# Initialize stacking classifier with meta-learner
stacking_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression())

pipeline_stacking = Pipeline([
    ('preprocessor', preprocessor),
    ('TomekLinks', TomekLinks()),
    ("scaler" , StandardScaler()),
    ('model', stacking_model )
])

pipeline_stacking.fit(X,y)




pickle.dump(pipeline_stacking, open('model.pkl', 'wb'))
# Save the column names instead of the entire X DataFrame
with open('input_columns.pkl', 'wb') as file:
    pickle.dump(X.columns.tolist(), file)