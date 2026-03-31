import streamlit as st
import requests

API_URL = 'https://empstatuslogistic.onrender.com/predict-status'

st.header("Employee Termination or Active Prediction using Logistic Regression")
st.subheader("Logisitc Regression Project")

# st.text("Performance Score (0-5)")
# st.text_input("Enter performance score", placeholder=3, value=3.0)

st.sidebar.header(
    "Employee Status Prediction"
)

PerfScoreID = st.sidebar.slider(
    "Performance Score",
    value=3,
    min_value=0,
    max_value=5,
    step=1 # default
)

Salary = st.sidebar.slider(
    "Salary",
    value=10000,
    min_value=1000,
    max_value=600000,
    step=10
)

PositionID = st.sidebar.slider(
    "Position ID",
    value=10,
    min_value=1,
    max_value=30,
    step=1
)

EngagementSurvey = st.sidebar.slider(
    "Engagement Survey",
    value=3.0,
    min_value=0.0,
    max_value=5.0,
    step=0.1
)

EmpSatisfaction = st.sidebar.slider(
    "Employee Satisfaction",
    value=3,
    min_value=0,
    max_value=5,
    step=1
)

SpecialProjectsCount = st.sidebar.slider(
    "Special Project Count",
    value=5,
    min_value=1,
    max_value=30,
    step=1
)

DaysLateLast30 = st.sidebar.slider(
    "Late Days",
    value=5,
    min_value=0,
    max_value=30,
    step=1
)

Absences = st.sidebar.slider(
    "Absences",
    value=5,
    min_value=0,
    max_value=30,
    step=1
)

if st.button("Predict Status"):
    payload = {
        "PerfScoreID": PerfScoreID,
        "Salary": Salary,
        "PositionID": PositionID,
        "EngagementSurvey": EngagementSurvey,
        "EmpSatisfaction": EmpSatisfaction,
        "SpecialProjectsCount": SpecialProjectsCount,
        "DaysLateLast30": DaysLateLast30,
        "Absences": Absences
    }

    try:
        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            result = response.json()

            if result["Predicted_Status"] == 1:
                st.success("Employee Likely to be Active in Future!!!")
            else:
                st.warning("Employee Likely to be Terminated in Future!!!")
        else:
            st.error("Error: API path is invalid!!")
    except requests.exceptions.RequestException:
        st.error("Could not connect to API!!")




    # # Feature Selection
    # input_data = pd.DataFrame([[
    #     PerfScoreID, Salary, PositionID, EngagementSurvey, EmpSatisfaction, SpecialProjectsCount,
    #     DaysLateLast30, Absences
    # ]], columns=features)
    # # Data Standarized
    # input_scaler = scaler.transform(input_data)
    # # Predict Data
    # prediction = model.predict(input_scaler)
    
    # if prediction[0] == 0:
    #     st.success("Employee is likely to be active in future🫳😌.")
    #     result = 'Active'
    # else:
    #     st.warning("Employee is likely to be terminated in future.😭😭😭")
    #     result = 'Terminated'

    # st.metric(
    #     "Employee Status",
    #     value=result
    # )

