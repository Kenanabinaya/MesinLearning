import pickle
import streamlit as st

# Memuat model yang telah di-train
model = pickle.load(open('estimasi_cars.sav', 'rb'))

st.title('Estimasi Harga Mobil di USA')

year = st.number_input('Input Tahun Mobil')
mileage = st.number_input('Input Jarak Tempuh mobil')
lot = st.number_input('Input Banyaknya Mobil')
model_input = st.number_input('Input Model')
color = st.number_input('Input Warna')
state = st.number_input('Input Negara')
condition = st.number_input('Input Kondisi')
vin = st.number_input('Input Vin')

prediksi = ''
if st.button('Estimasi Harga Mobil'):
    # Membuat array 2D untuk memasukkan ke model
    input_data = [[year, mileage, lot, model_input, color, state, condition, vin]]
    
    # Melakukan prediksi
    prediksi = model.predict(input_data)
    st.write('Estimasi Harga Mobil Dalam US : ', prediksi[0])
