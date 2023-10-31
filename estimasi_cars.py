import pickle
import streamlit as st

model = pickle.load(open('estimasi_cars.sav', 'rb'))

st.title('Estimasi Harga Mobil di USA')

year = st.number_input('Input Tahun Mobil')
mileage = st.number_input('Input Jarak Tempuh mobil')
lot = st.number_input('Input Banyak nya Mobil')

prediksi = ''
if st.button('Estimasi Harga Mobil'):
    prediksi = model.predict(
        [[year, mileage, lot]]
    )
    st.write('Estimasi Harga Mobil Dalam US : ', prediksi)
