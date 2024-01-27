# %%
import pandas as pd
import numpy as np
import sklearn
import pickle
import joblib
import streamlit as st

# %%
import shap
import time
from shap import summary_plot
from shap import TreeExplainer, Explanation
import streamlit.components.v1 as components

# %%
filename='RandomForest_model_ts_20.pkl'
model = open(filename, 'rb')
rf_model=joblib.load(model)

# %%
def rf_pred_prob(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall):
    columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg',
         'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']
    pred_arr=np.array([age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall])
    preds=pd.DataFrame(data=pred_arr.reshape(1,-1), columns=columns)
    model_prediction=rf_model.predict(preds)
    model_prediction_prob=rf_model.predict_proba(preds)
    return model_prediction, model_prediction_prob, preds    

# %%
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# %%
def run():

    st.title('CorMeum')
    st.title('XAI Model for Heart Attack Prediction using a Random Forest ML Model')
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
    cp=st.text_input("chest pain (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic)")
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
    probability=""
    preds_pd=pd.DataFrame()
    explainer=shap.TreeExplainer(rf_model)  # explainer
    shap_values =list()
    i=0
    
    if st.button("Predict"):
        i=1
        prediction=rf_pred_prob(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall)[0]
        probability=rf_pred_prob(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall)[1][0][1]*100
        preds_pd=rf_pred_prob(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall)[2]
        
        shap.initjs()
        shap_values = explainer.shap_values(preds_pd)
        


    st.success("Prediction: if your risk score is [0], it means you are not at risk of having a heart attack, and that's good news!, if it is [1] consult a specialist. Your risk value is: {}".format( prediction ))  
    st.success("Probability: Your probability risk value in % is: {}".format( probability ))
    st.subheader('your data at once :')
    st.write(preds_pd)  
    #st.info('Probability values: 0, class 1 or not risk attack and probabilities for each parameter.  1: class 2, or having a risk attack and probabilites for each parameter')
    #st.write(shap_values[1])
    st.write('Explainability (XAI): in pink are the parameters that increase the predicted probability risk, in blue those that decrease its value. In bold is the probability value of having a heart attack, as shown here:')
    if i==1:
        
        
        columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg',
         'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall']
        st.info('shap values for class  1, thus being a risk of a heart attack')
        st.write(pd.DataFrame(data=shap_values[0][0].reshape(1,-1), columns=columns))
        
        shap.initjs()
        st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1], preds_pd))

# %%
if __name__=='__main__':
    run()
