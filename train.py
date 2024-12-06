import pickle
from flask_app.ml_functions import (
    read_csv,
    preprocess_data,
    split_train_test,
    get_median,
    fill_missing_values,
    feature_engineere_all,
    select_model,
    train_model,
    categorical_cols,
    numerical_cols
)

df = read_csv('data/flight_train.csv')

clean_df = preprocess_data(df) 
# splitting
train_x, train_y, test_x, test_y = split_train_test(clean_df) 
# filling missing values (more cleaning)
median_dict = {col: get_median(train_x, col) for col in numerical_cols}
pickle.dump(median_dict, open('median_dict.pkl', 'wb')) # save it!
train_x = fill_missing_values(train_x, median_dict)
test_x = fill_missing_values(test_x, median_dict)
# feature engineering
train_x, train_y = feature_engineere_all(train_x, train_y, categorical_cols, is_train=True)
test_x, test_y = feature_engineere_all(test_x, test_y, categorical_cols, is_train=False)
# model selection and training
best_model = select_model(train_x, train_y, test_x, test_y) # model selection
best_model = train_model(best_model, train_x, train_y) # trained model