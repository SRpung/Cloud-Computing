import flask
import pandas as pd
import dash
from dash import dcc, html
# import plotly.express as px  # removed to reduce memory usage
import plotly.graph_objects as go
import plotly.io as pio
import pickle

# Disable Plotly default template to avoid memory heavy loading
pio.templates.default = None

# 1. Inisialisasi Server Flask Utama
server = flask.Flask(__name__)

# Memuat model dan scaler untuk prediksi
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Gagal memuat model/scaler: {e}")
    model = None
    scaler = None

# Rute Utama menggunakan file index.html dari folder templates
@server.route('/')
def index():
    return flask.render_template('index.html')

@server.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return flask.render_template('index.html', error="Model atau Scaler belum dilatih/tidak ditemukan.")
    
    try:
        # Ambil input data dari form
        pregnancies = int(flask.request.form.get('Pregnancies', 0))
        glucose = int(flask.request.form.get('Glucose', 0))
        blood_pressure = int(flask.request.form.get('BloodPressure', 0))
        skin_thickness = int(flask.request.form.get('SkinThickness', 0))
        insulin = int(flask.request.form.get('Insulin', 0))
        bmi = float(flask.request.form.get('BMI', 0.0))
        dpf = float(flask.request.form.get('DiabetesPedigreeFunction', 0.0))
        age = int(flask.request.form.get('Age', 0))
        
        # Buat DataFrame dengan urutan kolom yang sesuai saat training
        input_df = pd.DataFrame([[
            pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
        ]], columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Normalisasi input
        input_scaled = scaler.transform(input_df)
        
        # Lakukan prediksi
        pred = model.predict(input_scaled)[0]
        prediction = 'Diabetes' if pred == 1 else 'Sehat'
        
        # Kirim data kembali ke template untuk diisi ulang di form (agar user tidak ketik ulang)
        form_values = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': dpf,
            'Age': age
        }
        
        return flask.render_template('index.html', prediction=prediction, form_values=form_values)
    except Exception as e:
        return flask.render_template('index.html', error=f"Terjadi kesalahan saat memprediksi: {str(e)}")


# 2. Inisialisasi Dash Plotly dan dititipkan ke server Flask
app = dash.Dash(__name__, server=server, url_base_pathname='/dash/')

# Membaca data diabetes.csv untuk grafik dengan encoding yang benar
try:
    # 1. Tambahkan encoding='utf-8-sig' agar karakter aneh di judul kolom hilang
    df = pd.read_csv('diabetes.csv', encoding='utf-8-sig')
    
    # Bersihkan nama kolom dari spasi tidak terlihat (jika ada)
    df.columns = df.columns.str.strip()

    # 2. Lakukan grouping berdasarkan kolom 'Outcome' yang sudah bersih
    df_grouped = df.groupby('Outcome')['Glucose'].mean().reset_index()
    
    # 3. Mengubah angka 0 dan 1 menjadi label teks
    df_grouped['Outcome'] = df_grouped['Outcome'].map({0: 'Sehat', 1: 'Diabetes'})
    
    # 4. Membuat objek Grafik Batang
    import plotly.graph_objects as go
    # Create bar chart using Graph Objects for real data
    grouped_bar = go.Bar(x=df_grouped['Outcome'], y=df_grouped['Glucose'], marker_color=['#1f77b4', '#ff7f0e'])
    fig = go.Figure(data=[grouped_bar])
    fig.update_layout(title='Rata-rata Kadar Glukosa Berdasarkan Status Kesehatan', xaxis_title='Status', yaxis_title='Rata-rata Glukosa')
                 
except Exception as e:
    print(f"Terjadi error saat load data: {e}")
    # Create a simple placeholder figure using Plotly Graph Objects to avoid template loading issues
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Bar(x=["Apples", "Oranges", "Bananas"], y=[4, 1, 2])])
    fig.update_layout(title="Dummy Data", xaxis_title="Fruit", yaxis_title="Amount")

# 3. Atur Layout Halaman Dash (/dash/)
app.layout = html.Div(children=[
    html.H1(children='Menu Grafik Analisis (Dash Plotly)'),
    html.Div(children='Halaman ini berjalan di rute /dash/ pada server yang sama.'),
    dcc.Graph(
        id='example-graph',
        figure=fig
    ),
    html.Br(),
    html.A("Kembali ke Halaman Utama", href="/", style={'fontSize': '16px', 'color': '#3498db'})
])

# 4. Jalankan Aplikasi
if __name__ == '__main__':
    server.run(debug=True, use_reloader=False, port=5000)