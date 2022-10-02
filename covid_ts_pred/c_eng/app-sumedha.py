import streamlit as st

from covid_ts_pred.b_model.training import predict
from covid_ts_pred.b_viz.b_preproc import preprocessing

import matplotlib.pyplot as plt
st.write("Hello ,let's learn how to build a streamlit app together")
st.title ("Welcome to COVID-19 Prediction App !")
st.markdown( '''
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

('Brazil', 'France','Mexico','Spain'))


st.write('YOU SELECTED:', option)


country_list=['Brazil','France', 'Mexico','Spain']

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
    st.line_chart(data=[y_test.tolist(), list_pred], x=y_test.index.tolist())


fig, ax = plt.subplots(1, figsize=(17,7))
plt.plot(y_test.index, list_pred,color='r');
plt.plot(y_test.index, y_test);
ax.set_title("Covid 19 calculation for different countries", size=10)
ax.set_ylabel("Number of death cases", size=10)
ax.set_xlabel("Date", size=13)
st.pyplot(fig)
