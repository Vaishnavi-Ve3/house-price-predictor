from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model/house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html', form_data={}, prediction_text='')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = {
            'bedrooms': request.form['bedrooms'],
            'bathrooms': request.form['bathrooms'],
            'living_area': request.form['living_area'],
            'lot_area': request.form['lot_area'],
            'floors': request.form['floors'],
            'waterfront': request.form['waterfront'],
            'views': request.form['views'],
            'condition': request.form['condition'],
        }

        data = [float(form_data[key]) for key in form_data]
        prediction = model.predict([data])[0]

        return render_template(
            'index.html',
            prediction_text=f'Estimated House Price: ₹{int(prediction):,}',
            form_data=form_data
        )
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}', form_data=request.form)


if __name__ == '__main__':
    app.run(debug=True)
