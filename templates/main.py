from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form
    pclass = int(request.form['pclass'])
    sex = request.form['sex']
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])
    fare = float(request.form['fare'])

    # Tạo DataFrame từ dữ liệu nhập vào
    data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
    })

    # Tiền xử lý dữ liệu
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)

    # Dự đoán số người sống sót
    prediction = model.predict(data)

    # Hiển thị kết quả
    if prediction[0] == 0:
        result = 'Không sống sót'
    else:
        result = 'Sống sót'

    return render_template('index.html', prediction_text='Dự đoán: {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)
