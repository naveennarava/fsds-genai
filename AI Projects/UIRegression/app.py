from flask import Flask, render_template,request
import pandas as pd
import pickle as pk
app = Flask(__name__)
model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet', 
    'PolynomialRegression', 'SGDRegressor', 'ANN', 'RandomForest', 'SVM', 'LGBM', 
    'XGBoost', 'KNN'
]
models={name:pk.load(open(f'{name}.pk1','rb')) for name in model_names}
results_df=pd.read_csv('model_evaluation_results.csv')

@app.route("/")
def index():
    return render_template('index.html',model_names=model_names)

@app.route("/predict",methods=['POST'])
def predictions():
    model_name=request.form['model']
    inputdata={
        'Avg. Area Income': float(request.form['Avg. Area Income']),
        'Avg. Area House Age': float(request.form['Avg. Area House Age']),
        'Avg. Area Number of Rooms': float(request.form['Avg. Area Number of Rooms']),
        'Avg. Area Number of Bedrooms': float(request.form['Avg. Area Number of Bedrooms']),
        'Area Population': float(request.form['Area Population'])
    }
    inputdf=pd.DataFrame([inputdata])
    if model_name in models:
       regresion= models[model_name]
       prediction= regresion.predict(inputdf)[0]
       return render_template('results.html', prediction=prediction, model_name=model_name)
    else:
        return jsonify({'error': 'Model not found'}), 400

@app.route('/results')
def results():
    return render_template('model.html', tables=[results_df.to_html(classes='data')], titles=results_df.columns.values)

if __name__ == '__main__':
    app.run(debug=True)

    