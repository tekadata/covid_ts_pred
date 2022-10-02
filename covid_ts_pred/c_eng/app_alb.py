import os
import pandas as pd
import streamlit as st
from covid_ts_pred.b_model.training import predict
from covid_ts_pred.b_viz.b_preproc import preprocessing

primaryColor="#F63366"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F0F2F6"
textColor="#262730"
font="sans serif"

st.markdown(
    '''
    #        HELLO !
    '''
    '''
    #        Welcome to COVID-19 Prediction App !
    '''



    '''
    ##### Our COVID-19 prediction models for TOTAL NUMBER OF DEATHS on next 10 days were based on restriction indicators, stringency index, and vaccination campaigns



    They were performed by Alberto, Sumedha, Thomas, and Kim under supervision of Arnaud and TAs


    ''')



option=st.selectbox('PLEASE SELECT YOUR COUNTRY',

('Brazil', 'France', 'India', 'Mexico', 'United Kingdom'))


st.write('YOU SELECTED:', option)



# if country_code = 'Brazil':
# if country_code = 'France'
# if country_code = 'India'
# if country_code = 'Mexico'
# if country_code = 'United Kingdom'

# prediction_window = 1st to 10th september 2022

country_list=['Brazil','France', 'India', 'Mexico', 'United Kingdom']
countries=[]


#path_begin= f'data/out_csv/index_{option}.csv'

# csv_name = f'index_{option}.csv'
# csv_path = os.path.join(path_begin, csv_name)


date_prediction=['2022/09/01', '2022/09/02', '2022/09/03', '2022/09/04', '2022/09/05'
'2022/09/06', '2022/09/07', '2022/09/08', '2022/09/09', '2022/09/10']


if __name__ == "__main__":
    # 1. Load Data
    # df = load_data(DATASET_PATH)

    # 2. Data prepartion
    X_test, y_test, X_train, y_train, df, y = preprocessing(option.capitalize(), n_days=len(date_prediction))

    # 3. Prediction
    list_pred, X_predict = predict(X_test=X_test, country=option.capitalize(), data=df, y=y, n_days=len(date_prediction))

    # 4. Plot the predicted and real total deaths by COVID-19
    st.line_chart(data=[y_test, list_pred], x=date_prediction)

# find csv
# read csv
# preprocess csv with Sumedha process
# predict the X_predict(dataframe)
