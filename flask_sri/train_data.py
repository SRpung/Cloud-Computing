import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Membaca data mentah dari file CSV Anda
data = pd.read_csv('diabetes.csv')

# 2. Memisahkan Fitur (X) dan Target (y)
X = data.drop(columns=['Outcome'])
y = data['Outcome']

# 3. Membagi data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. MEMBUAT DAN MELATIH SCALER (Normalisasi Angka)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Scaler belajar dari data training
X_test_scaled = scaler.transform(X_test)

# 5. Melatih Model Machine Learning dengan data yang sudah di-scale
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# 6. MENYIMPAN FILE scaler.pkl
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# 7. MENYIMPAN FILE model.pkl
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Berhasil! File 'scaler.pkl' dan 'model.pkl' telah dibuat di folder Anda.")