# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 01:36:28 2020

@author: Hardik
"""


import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

from datetime import datetime
import re


from sklearn.feature_extraction.text import CountVectorizer   #for BOW
from sklearn.feature_extraction.text import TfidfVectorizer   #for tfidf

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import SGDClassifier
import streamlit as st
import pickle

#Load the pickled models
minmax_model = pickle.load(open("minmax_scaler.pkl", 'rb'))
pca_model = pickle.load(open('pca_model.pkl', 'rb'))
DT_model = pickle.load(open('Decision_tree.pkl', 'rb'))
RF_model = pickle.load(open('Random_Forest.pkl', 'rb'))


def read_data():
    
    data = pd.read_csv('eda_data.csv')
    data.drop(["Unnamed: 0"], axis = 1, inplace = True)
    
    return data

def input_data():
    #Input the values for Prediction.
        
        age = st.number_input('Age:')
        distancefromhome = st.number_input('Distance From Home:')
        monthlyincome = st.number_input('Monthly Income:')
        nofcompaniesworked = st.number_input('Number of Companies Employee has worked for?')
        percentsalaryhike = st.number_input('Percent Salary Hike:')
        totalworkyears = st.number_input('Total Working Years:')
        yearsatcompany = st.number_input('How many years is the employee working in the Company?')
        trainingtimeslastyear = st.number_input('Number of Times gone under Training Last Year?')
        yearsincelastpromotion = st.number_input('How many years before the employee had last promotion?')
        yearswithcurrmanager = st.number_input('How many years have been with current Manager?')
        
        
        
        businesstravel = st.radio('How often does the Employee Travel for Business?',options = ['','Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
        department = st.radio('What is the Department of Employee?',options = ['','Research & Development', 'Sales', 'Human Resources'])
        education = st.radio('What is the Education of Employee?',options = ['','1', '2', '3', '4', '5'])
        educationfield = st.radio('What is the Field of Education of Employee?',options = ['','Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources',
                                                                                     'Other'])
        gender = st.radio('What is the Gender of Employee?',options = ['','Male', 'Female'])
        joblevel = st.radio('What is the Seniority/Level of current job?',options = ['','1', '2', '3', '4', '5'])
        jobrole = st.radio('What is the Role at current job?',options = ['','Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                                                                   'Healthcare Representative','Manager','Sales Representative','Research Director',
                                                                   'Human Resources'])
        maritalstatus = st.radio('What is Employees Marital Status?',options = ['','Married', 'Single', 'Divorced'])
        stockoptionlevel = st.radio('What is Employees Stock Option Level?',options = ['','0', '1', '2', '3'])        
        environmentsatisfaction = st.radio('What is Employees Environment Satisfaction Level?',options = ['','1.0', '2.0', '3.0', '4.0'])        
        jobsatisfaction = st.radio('What is Employees Job Satisfaction Level?',options = ['', '1.0', '2.0', '3.0', '4.0'])
        worklifebalance = st.radio('What is Employees Work-Life Balance?',options = ['','1.0', '2.0', '3.0', '4.0'])
        jobinvolvement = st.radio('What is Employees Job Involvement Level??',options = ['','1', '2', '3', '4'])
        performancerating = st.radio(label = 'What is Employees Performance Rating??',options = ['','3', '4'])
        
        
        dict_var = {'BusinessTravel' : businesstravel,
                    'Department' : department, 
                    'Education' : education, 
                    'EducationField' : educationfield,
                    'Gender' : gender, 
                    'JobLevel' : joblevel, 
                    'JobRole' : jobrole, 
                    'MaritalStatus' : maritalstatus,
                    'StockOptionLevel' : stockoptionlevel,
                    'EnvironmentSatisfaction' : environmentsatisfaction, 
                    'JobSatisfaction' : jobsatisfaction,
                    'WorkLifeBalance' : worklifebalance,
                    'JobInvolvement' : jobinvolvement, 
                    'PerformanceRating' : performancerating}
        
        
        cont_var = [age,distancefromhome,monthlyincome,nofcompaniesworked,percentsalaryhike,totalworkyears,yearsatcompany,
                    trainingtimeslastyear,yearsincelastpromotion,yearswithcurrmanager]
        
        #varib = [businesstravel,department,education,educationfield,gender,joblevel,jobrole,maritalstatus,stockoptionlevel,
                # environmentsatisfaction,jobsatisfaction,worklifebalance,jobinvolvement,performancerating]
        
        return cont_var, dict_var


def preprocessing(cont_var, dict_var):
    #Creating  dataframe
        
        
        
        all_list = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
                       'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
                       'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                       'BusinessTravel_Non-Travel', 'BusinessTravel_Travel_Frequently',
                       'BusinessTravel_Travel_Rarely', 'Department_Human Resources',
                       'Department_Research & Development', 'Department_Sales', 'Education_1',
                       'Education_2', 'Education_3', 'Education_4', 'Education_5',
                       'EducationField_Human Resources', 'EducationField_Life Sciences',
                       'EducationField_Marketing', 'EducationField_Medical',
                       'EducationField_Other', 'EducationField_Technical Degree',
                       'Gender_Female', 'Gender_Male', 'JobLevel_1', 'JobLevel_2',
                       'JobLevel_3', 'JobLevel_4', 'JobLevel_5',
                       'JobRole_Healthcare Representative', 'JobRole_Human Resources',
                       'JobRole_Laboratory Technician', 'JobRole_Manager',
                       'JobRole_Manufacturing Director', 'JobRole_Research Director',
                       'JobRole_Research Scientist', 'JobRole_Sales Executive',
                       'JobRole_Sales Representative', 'MaritalStatus_Divorced',
                       'MaritalStatus_Married', 'MaritalStatus_Single', 'StockOptionLevel_0',
                       'StockOptionLevel_1', 'StockOptionLevel_2', 'StockOptionLevel_3',
                       'EnvironmentSatisfaction_1.0', 'EnvironmentSatisfaction_2.0',
                       'EnvironmentSatisfaction_3.0', 'EnvironmentSatisfaction_4.0',
                       'JobSatisfaction_1.0', 'JobSatisfaction_2.0', 'JobSatisfaction_3.0',
                       'JobSatisfaction_4.0', 'WorkLifeBalance_1.0', 'WorkLifeBalance_2.0',
                       'WorkLifeBalance_3.0', 'WorkLifeBalance_4.0', 'JobInvolvement_1',
                       'JobInvolvement_2', 'JobInvolvement_3', 'JobInvolvement_4',
                       'PerformanceRating_3', 'PerformanceRating_4']    
    
    
        df = pd.DataFrame([np.zeros(68)], columns = all_list)
        
        #cont_var = [age,distancefromhome,monthlyincome,nofcompaniesworked,percentsalaryhike,totalworkyears,yearsatcompany,
                    #trainingtimeslastyear,yearsincelastpromotion,yearswithcurrmanager]
        cont_all_list = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'NumCompaniesWorked',
       'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
        
        for i, j in zip(cont_all_list, cont_var):
            df[i] = j
        
        
        #varib #= [businesstravel,department,education,educationfield,gender,joblevel,jobrole,maritalstatus,stockoptionlevel,environmentsatisfaction,jobsatisfaction,worklifebalance,jobinvolvement,performancerating]        
        
        catvar = ['BusinessTravel',
       'Department', 'Education', 'EducationField',
       'Gender', 'JobLevel', 'JobRole', 'MaritalStatus',
       'StockOptionLevel',
       'EnvironmentSatisfaction', 'JobSatisfaction','WorkLifeBalance',
       'JobInvolvement', 'PerformanceRating']
        
        

        
        for i in catvar:
            df[(i+'_'+str(dict_var[i]))] = 1
                    
                    
        
        return df
    
    
def pred(df):
    
    sx = minmax_model.transform(df)
    sx = pca_model.transform(sx)
    prediction = DT_model.predict(sx)
    return prediction
    

def main():
    "ML APP"
    
    st.title("Employee Attrition Web App")
    st.subheader("Built with ML")
    
    #Menu
    menu = ["Data Preview", "Prediction"]
    choices = st.sidebar.selectbox("Select Activities", menu)
    
    if choices == 'Data Preview':
        st.subheader("Data Preview")
        
        data = read_data()
        
        st.write("Do you want to preview the whole dataset?")
        if st.button("Yes"):
            st.write(data)
            
        st.write("Enter the number of records you want to see from top:")
        head_n = st.number_input("---")
        if head_n != 0:    
            st.write(data.head(int(head_n)))
            
        st.write("Enter the number of records you want to see from bottom:")
        tail_n = st.number_input("----")
        if tail_n != 0:    
            st.write(data.tail(int(tail_n)))
            
        st.write("Shape of the dataset:")
        if st.checkbox("Show Shape"):
            st.write(data.shape)
            
        st.write("Show the Statistics about the dataset.")
        if st.checkbox("Description"):
            st.write(data.describe())
        
        if st.checkbox("Information"):
            st.write(data.info)

    elif choices == 'Prediction':
        #df = input_data()
        cont_var, dict_var = input_data()

        dff = preprocessing(cont_var, dict_var)
        if st.button("Submit"):
           st.dataframe(dff)

           sx = pred(dff)
           st.write(sx)

           if sx == 1:
                st.write("This Employee will leave the company.")
           else:
                st.write("This Employee will not leave the company.")



if __name__ == '__main__':
    main()