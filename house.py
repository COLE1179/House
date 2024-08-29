import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



st.header('House Prices Prediction Created By Cole')

dg = pd.read_csv("Bengaluru_House_Data.csv")
st.dataframe(dg)

df = dg[['area_type', 'location', 'availability',  'bath', 'balcony', 'price']].dropna()

encoder_area_type = LabelEncoder()
df['area_type'] = encoder_area_type.fit_transform(df['area_type'])

encoder_location = LabelEncoder()
df['location'] = encoder_location.fit_transform(df['location'])

encoder_availability = LabelEncoder()
df['availability'] = encoder_availability.fit_transform(df['availability'])

x = df[['area_type', 'location', 'availability', 'bath', 'balcony']]
y = df[['price']]

#20% of the dataset is for testing and 70% of the dataset is for training
feature_train, feature_test, target_train, target_test = train_test_split(x, y, test_size=0.2)

model = LinearRegression()
model.fit(feature_train, target_train)

#student portal features which will be written on the sidebar
st.sidebar.title('House prices features')
st.sidebar.header('Please put your information')

area_type =st.sidebar.selectbox('area_type', encoder_area_type.classes_)
location = st.sidebar.selectbox('location', encoder_location.classes_)
availability = st.sidebar.selectbox('availability', encoder_availability.classes_)
bath = st.sidebar.slider('Number of Bath', 1, 10, 2)
balcony = st.sidebar.slider('Number of Balcony', 1, 10, 2)

area_type_encoded = encoder_area_type.transform([area_type])[0]
location_encoded = encoder_location.transform([location])[0]
availability_encoded = encoder_availability.transform([availability])[0]

total = {
         'area_type': [area_type_encoded],
         'location': [location_encoded],
         'availability': [availability_encoded],
         'bath': [bath],
         'balcony': [balcony]}

#print(total)
st.write('House Details')
st.dataframe(total, width=700)
pf = pd.DataFrame(total)



if st.button('Check Prediction'):
    prediction = model.predict(pf)
    #st.write('The price of the house is, $', prediction[0])
    st.write(f'The price of the house is: $ {prediction[0][0]:,.2f}')


