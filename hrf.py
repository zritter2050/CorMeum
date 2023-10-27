# %%
import pandas as pd
import numpy as np
import sklearn
import pickle
import joblib
import streamlit as st

# %%
filename='RandomForest_model_ts_20.pkl'
model = open(filename, 'rb')
rf_model=joblib.load(model)

# %%
def rf_prediction(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall):
    pred_arr=np.array([age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall])
    preds=pred_arr.reshape(1,-1)
    model_prediction=rf_model.predict(preds)
    return model_prediction    

# %%
def run():
    st.title('CorMeum')
    st.title('Model for Heart Attack Prediction (RFM)')
    st.text('Author: Dr. Z. Ritter')
    st.text('Model accuracy = 0.836')
    st.text('Highest 5 feature importances: rate of influence of item on prediction (scroll bar below)')
    st.text('chest pain (0.17), ST depression induced by exercise (0.117), thalassemia (0.113), maximun heart rate (0.104), cholesterol (0.08)')
    
    # show data frame feature importances sorted
    if st.checkbox('Show dataframe (all feature rate importances)'):
        file="FeatImportancesRF_ts_20.csv"
        chart_data = pd.DataFrame(pd.read_csv(file,sep=',',))
        chart_data
        
    if st.checkbox('Show features description'):
        file="HeartFeaturesDescription.csv"
        chart_data = pd.DataFrame(pd.read_csv(file,sep=',',))
        chart_data
       

    html_temp=""
    ""
    st.markdown(html_temp)
    
    age=st.text_input("enter age")
    sex=st.text_input("sex, 0 = female, 1 = male")
    cp=st.text_input("chest pain: 0:typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic")
    trtbps=st.text_input("resting blood pressure (mm Hg on admission to the hospital)")
    chol=st.text_input("cholesterol in mg/dl")
    fbs=st.text_input("fasting blood sugar (> 120 mg/dl, 1 = true, 0 = false)")
    restecg=st.text_input("resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)")
    thalachh=st.text_input("maximum heart rate achieved")
    exng=st.text_input("exercise induced angina (1 = yes; 0 = no)")
    oldpeak=st.text_input("ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot)") 
    slp=st.text_input("slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping")
    caa=st.text_input("number of major vessels (0-3)") 
    thall=st.text_input("on thalassemia blood disorder (1 = normal, 2 = fixed defect, 3 = reversable defect)")
    
    prediction=""
    if st.button("Predict"):
        prediction=rf_prediction(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall)
    st.success("Prediction: if your risk score is [0], it means you are not at risk of having a heart attack, and that's good news!, if it is [1] consult a specialist. Your risk value is: is {}".format( prediction ))  


# %%
if __name__=='__main__':
    run()