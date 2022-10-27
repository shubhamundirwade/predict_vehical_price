
import streamlit as st
import numpy as np
import xgboost as xgb
import json

st.cache(allow_output_mutation= True)

def json_file():
    with open("columns.json") as columns:
        data_json = json.loads(columns.read())
        data_json = np.asarray(data_json["data_columns"])

    return data_json

# user inputs 
st.cache(allow_output_mutation=True)

def UserInputs():
    manufacturer = st.selectbox("Manufacturer",("Ford","Toyota","Hyundai",
                                                "Volkswagen","Skoda","Vauxhall",
                                                "BMW","Audi","Mercedes-Benz"))
    if manufacturer == "Ford":
        model_car = st.selectbox("Model",('Fiesta', 'Focus', 'Kuga', 'EcoSport', 'C-MAX', 'Ka+',
       'Tourneo Custom', 'S-MAX', 'B-MAX', 'Edge', 'Tourneo Connect',
       'Puma', 'Mondeo', 'KA', 'Grand C-MAX', 'Galaxy', 'Mustang',
       'Grand Tourneo Connect', 'Fusion'))
    if manufacturer == "Toyota":
        
        
        model_car = st.selectbox("Model",('GT86', 'Corolla', 'RAV4', 'Yaris', 'Auris', 'Aygo', 'C-HR',
       'Prius', 'Avensis', 'Verso', 'Hilux', 'Land Cruiser', 'Camry'))
    
        
    if manufacturer == "Hyundai":
        
        model_car = st.selectbox("Model",('I20', 'Tucson', 'I10', 'IX35', 'I30', 'I40', 'Ioniq', 'Kona',
       'I800', 'IX20', 'Santa Fe'))
        
        
    if manufacturer == "Volkswagen":
        model_car = st.selectbox("Model",('T-Roc', 'Golf', 'Passat', 
                                          'T-Cross', 'Polo', 'Tiguan', 'Sharan',
                                           'Up', 'Scirocco', 'Beetle', 
                                          'Caddy Maxi Life', 'Caravelle',
                                           'Touareg', 'Arteon', 
                                          'Touran', 'Golf SV', 'Amarok',
                                           'Tiguan Allspace', 'Shuttle',
                                          'Jetta', 'CC', 'California'))
        
    if manufacturer == "Skoda":
        model_car = st.selectbox("Model",('Octavia', 'Yeti Outdoor', 
                                          'Superb', 'Rapid', 'Karoq', 'Fabia',
                                        'Yeti', 'Kodiaq', 'Scala', 
                                          'Citigo', 'Roomster', 'Kamiq'))
                                         
    
    
    if manufacturer == "Vauxhall":
        model_car = st.selectbox("Model",('Corsa', 'Astra', 'Viva', 'Mokka', 'Mokka X', 'Crossland X',
       'Zafira', 'Meriva', 'Zafira Tourer', 'Adam', 'Grandland X',
       'Antara', 'Insignia', 'GTC', 'Combo Life', 'Vivaro', 'Agila')) 
        
        
        
    if manufacturer == "BMW":
        
        model_car = st.selectbox("Model",('5 Series', '6 Series', '1 Series', '7 Series', '2 Series',
       '4 Series', 'X3', '3 Series', 'X5', 'X4', 'X1', 'M4', 'X6', 'Z4',
       'X2', 'i8', 'M2', 'i3', '8 Series', 'M3', 'M5'))
        
        
    if manufacturer == "Audi":
        
        model_car = st.selectbox("Model",('A1', 'A6', 'A4', 'A3', 'Q3', 'Q5', 'A5', 'Q2', 'A7', 'RS6', 'Q7',
       'A8', 'TT', 'Q8', 'RS4', 'RS5', 'RS3', 'R8', 'SQ5', 'S3'))
        
        
    if manufacturer == "Mercedes-Benz":
        
        model_car = st.selectbox("Model",('SL CLASS', 'GLE Class', 'GLA Class', 'GLC Class', 'B Class',
       'C Class', 'E Class', 'GL Class', 'CLS Class', 'A Class', 'SLK',
       'CLA Class', 'V Class', 'CL Class', 'GLS Class', 'M Class',
       'X-CLASS', 'S Class'))

    trasmission = st.radio('Trasmission',('Automatic','Manual','Semi-Auto'))
    
    year = st.slider('Year',min_value = 2000,max_value = 2020,step = 1)
    
    
    fuelType = st.radio('Fuel Type',('Diesel','Hybrid','Petrol'))
    
    engineSize = st.selectbox('Engine Size',(0.0, 1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,2,2.1,2.2,
       2.3,2.4,2.5,2.8,2.9,3.0,3.2,3.5,4.0,4.2,4.4,4.7,5.0,5.2, 5.5, 6.2))
   
     
    mileage = st.number_input('Mile Age',min_value = 3100,max_value = 100000)
   

    return manufacturer,model_car,trasmission,year,fuelType,engineSize, mileage

    

st.cache(allow_output_mutation=True)

def preprocess():
    
    columns = json_file()
    model_car,manufacturer,trasmision,year,fuelType,engineSize,mileage= UserInputs()
    
    data = np.zeros(len(columns))
    
    model_idx = np.where(columns == model_car)[0][0]
    trasmision_idx = np.where(columns == trasmision)[0][0]
    fuel_type_idx = np.where(columns == fuelType)[0][0]
    
    data[165] = year
    data[170] = engineSize
    data[171] = mileage

    
    if model_idx >=0:
        data[model_idx] = 1
        
    if trasmision_idx >=0:
        data[trasmision_idx] = 1
        
    if fuel_type_idx >=0:
        data[fuel_type_idx] = 1
        
    return np.asarray([data])

    
st.cache(allow_output_mutation=True)

def predict(new_data):
    model = xgb.XGBRegressor().load_model("cars_sales_model.json")
    return  np.round(model.predict(new_data)).astype(int)


st.cache(allow_output_mutation=True)

def main():
    
    st.subheader("User Input")
    new_data = preprocess()
    if st.button(label = 'Predict'):
        
        price=predict(new_data)
        st.success(f'The estimated price of the vehicle is: $ {price} £')
