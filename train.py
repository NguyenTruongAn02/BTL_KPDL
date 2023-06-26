import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import pickle

# Đường dẫn đến file CSV
csv_file = 'train.csv'

# Đọc dữ liệu từ file CSV
data = pd.read_csv(csv_file)

# Tiền xử lý dữ liệu
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].mean(), inplace=True)

le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

# Lựa chọn các cột phù hợp và chia dữ liệu thành features (X) và target (y)
selected_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = data[selected_columns]
y = data['Survived']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tiến hành huấn luyện mô hình
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm tra (tuỳ chọn)
accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)

# Lưu mô hình vào file model.pkl
pickle.dump(model, open('model.pkl', 'wb'))
