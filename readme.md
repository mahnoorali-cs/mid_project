Project Title: 
        Patient Disease Risk Prediction using ML Models

Goal: 
    The goal is to predict the treatment sucess rate on the basis of patient medical data, predicting the diease is present or not, Natural Language Processing (NLP) to analyze the medical_report_text written by doctors to classify into mild, moderate and severe conditions, and categorizing patients into high, medium and low risk and predicting the class on the basis of medical and lifestyle factors.

Dataset Info:
        Age  __ in integers <br>
        Gender  __ Male or Female <br>
        Blood_pressure  __ format: 120/80 <br>
        Cholesterol_level  __ in integers <br>
        BMI  __ in float <br>
        Smoking_habit  __Yes/No <br>
        Exercise_frequency  __ per week in float <br>
        Glucose_level  __in integers <br>
        Medical_report_text  __Doctor’s notes and patient symptoms <br>
        Disease_present   __Yes/No<br>
        Treatment_success_rate   __0–100% <br>

Tech Stack:
        Numpy, Pandas, Matplotlib, Scikit-learn

Models used:
        Linear Regression, Logistic Regerssion, Naive Bayes, SVM

Evaluation Metrics:
            Mean_Squared_Error, R2_score, Accuracy, Recall, Precision, F1_score, Confusion Matrix,
            Classification Report
            
Running the interface: <br>
           1. clone the repo <br>
           2. go into the folder <br>
           3. install the requirments <br>
           4. streamlit run app.py <br>

Results Summary:
            After all preprocessing the algorithms are applied, evaluated and visualized performance. The accuracy of SVM is most then naive bayes and then logistic regression. 

    
