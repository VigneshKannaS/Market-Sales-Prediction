from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define a function to preprocess the input data
def preprocessInput(inputList):
    inputArray = np.array(inputList, dtype=np.float32) #converting list of elements into an array
    return inputArray.reshape(1, -1) #returning reshaped data

@app.route('/')
def home():
    return render_template('index.html') #rendering index.html

@app.route('/predict', methods=['POST'])
def predict():
    itemIdentifier = request.form['itemIdentifier']
    itemWeight = request.form['itemWeight']
    itemFatContent = request.form['itemFatContent']
    itemVisibility = request.form['itemVisibility']
    itemType = request.form['itemType']
    itemMRP = request.form['itemMRP']
    outletIdentifier = request.form['outletIdentifier']
    outletEstablishmentYear = request.form['outletEstablishmentYear']
    outletSize = request.form['outletSize']
    outletLocationType = request.form['outletLocationType']
    outletType = request.form['outletType']

    inputList = itemIdentifier+','+itemWeight+','+itemFatContent+','+itemVisibility+','+itemType+','+itemMRP+','+outletIdentifier+','+outletEstablishmentYear+','+outletSize+','+outletLocationType+','+outletType
    inputList = inputList.split(',')
    inputArray = preprocessInput(inputList)
    prediction = model.predict(inputArray)[0]
    outputMessage = f'The predicted sales value is {prediction:.2f} in Indian rupees'
    return render_template('index.html', predictionText=outputMessage)
if __name__ == '__main__':
    app.run(debug=True)
