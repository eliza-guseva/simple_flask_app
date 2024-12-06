from flask import Flask, render_template, request, Response
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ml_functions import (
    preprocess_data,
    fill_missing_values,
    feature_engineere_all,
    read_csv,
    split_train_test,
    get_median,
    select_model,
    train_model,
    categorical_cols,
    numerical_cols
)
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_train', methods=['POST'])
def upload_train():
    try:
        file = request.files['file']
        if not file:
            return "<div class='error'>No file uploaded</div>", 400
        
        # First save the uploaded file
        uploaded_file_path = 'train_data.csv'
        file.save(uploaded_file_path)
        return "<div class='success'>Training data uploaded successfully. Please click 'Train Model' to proceed.</div>", 200
        
    except Exception as e:
        print(f"Error in upload_train: {e}")
        return f"<div class='error'>Error: {str(e)}</div>", 400

@app.route('/train', methods=['POST'])
def train():
    logger.info("Training model")
    with open('status.txt', 'w') as f:
        f.write("<div class='progress-message'>Began training model...</div>")
    try:
        if not os.path.exists('train_data.csv'):
            return "<div class='error'>Please upload training data first</div>", 400
            
        # Add progress messages
        progress_html = "<div class='progress-message'>Loading and preprocessing data...</div>"
        with open('status.txt', 'w') as f:
            f.write(progress_html)
        logger.info("Reading training data")
        df = read_csv('train_data.csv')
        clean_df = preprocess_data(df)
        
        progress_html = "<div class='progress-message'>Splitting data and handling missing values...</div>"
        with open('status.txt', 'w') as f:
            f.write(progress_html)
        # Split data
        train_x, train_y, test_x, test_y = split_train_test(clean_df)
        
        # Fill missing values
        median_dict = {col: get_median(train_x, col) for col in numerical_cols}
        pickle.dump(median_dict, open('median_dict.pkl', 'wb'))
        train_x = fill_missing_values(train_x, median_dict)
        test_x = fill_missing_values(test_x, median_dict)
        
        progress_html = "<div class='progress-message'>Performing feature engineering...</div>"
        with open('status.txt', 'w') as f:
            f.write(progress_html)
        # Feature engineering
        train_x, train_y = feature_engineere_all(train_x, train_y, categorical_cols, is_train=True)
        test_x, test_y = feature_engineere_all(test_x, test_y, categorical_cols, is_train=False)
        
        progress_html = "<div class='progress-message'>Training and selecting best model...</div>"
        with open('status.txt', 'w') as f:
            f.write(progress_html)
        best_model, best_params, best_score = select_model(train_x, train_y, test_x, test_y)
        best_model = train_model(best_model, train_x, train_y)
        
        pickle.dump(best_model, open('best_model.pkl', 'wb'))
        
        final_message = f"<div class='success'>Model trained successfully! Best model accuracy: {best_score:.2f}</div>"
        with open('status.txt', 'w') as f:
            f.write(final_message)
        return final_message, 200
    except Exception as e:
        print(e)
        return f"<div class='error'>Error: {str(e)}</div>", 400

@app.route('/upload_test', methods=['POST'])
def upload_test():
    try:
        # Log the incoming request
        logger.info("Received upload_test request")
        
        file = request.files['file']
        logger.info(f"File received: {file.filename}")
        
        if not file:
            logger.error("No file uploaded")
            return "<div class='error'>No file uploaded</div>", 400
        
        # Check if model exists
        if not os.path.exists('best_model.pkl'):
            logger.error("Model file (best_model.pkl) not found")
            return "<div class='error'>Please train the model first</div>", 400
            
        if not os.path.exists('median_dict.pkl'):
            logger.error("Median dictionary (median_dict.pkl) not found")
            return "<div class='error'>Please train the model first</div>", 400

        # Save the uploaded test file
        uploaded_file_path = 'test_data.csv'
        file.save(uploaded_file_path)
        logger.info(f"File saved to {uploaded_file_path}")
        
        # Load the model and median dictionary
        logger.info("Loading model and median dictionary")
        model = pickle.load(open('best_model.pkl', 'rb'))
        median_dict = pickle.load(open('median_dict.pkl', 'rb'))
        
        # Process test data
        logger.info("Processing test data")
        df = read_csv(uploaded_file_path)
        if 'satisfaction' in df.columns:
            df = df.drop(columns=['satisfaction'])
        clean_df = preprocess_data(df)
        test_x = fill_missing_values(clean_df, median_dict)
        test_x, _ = feature_engineere_all(test_x, None, categorical_cols, is_train=False)
        
        # Make predictions
        logger.info("Making predictions")
        
        predictions = model.predict(test_x)
        
        # Save predictions to CSV
        predictions_df = pd.DataFrame(predictions, columns=['prediction'])
        predictions_df.to_csv('predictions.csv', index=False)
        logger.info("Predictions saved to predictions.csv")
        
        return "<div class='success'>Predictions generated successfully! Check predictions.csv file.</div>", 200
        
    except Exception as e:
        logger.error(f"Error in upload_test: {e}", exc_info=True)
        return f"<div class='error'>Error: {str(e)}</div>", 400

@app.route('/progress')
def progress():
    def generate():
        logger.info("SSE connection established")  # Log when connection starts
        while True:
            try:
                with open('status.txt', 'r') as f:
                    content = f.read()
                    logger.info(f"Read from status.txt: {content}")  # Log what we're reading
                    if content:  # Only yield if there's content
                        yield f"data: {content}\n\n"
                        
                time.sleep(30)
            except Exception as e:
                logger.error(f"Error in SSE generate: {e}", exc_info=True)
                break
        logger.info("SSE connection closed")  # Log when connection ends
    
    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'  # Needed for some NGINX setups
    return response

if __name__ == '__main__':
    app.run(debug=True)

