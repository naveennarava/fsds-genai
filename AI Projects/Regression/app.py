import streamlit as st 
import pickle as pk
import numpy as np
model = pk.load(open(r'C:\AI Projects\Regression\linermodel.pk1', 'rb'))
st.title('Salary Prediction')
st.write('this a prdiction salary')
yearsexp=st.number_input("Enter no of years",min_value=0.0,max_value=50.0,value=1.0,step=0.5)
if st.button("PredictSalary"):
    exp=np.array([[yearsexp]])
    pred= model.predict(exp)
    
     # Display the result
    st.success(f"The predicted salary for {yearsexp} years of experience is: ${pred[0]:,.2f}")
    
# Display information about the model
st.write("The model was trained using a dataset of salaries and years of experience.")

    



