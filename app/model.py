# Machine Learning Model
import pandas as pd
import joblib # to convert model file to binary file (pkl file)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('D:/Broadway/Data Science - 2/fastapi/data/HR_Dataset Refresh.csv')

MODEL_PATH = 'D:/Broadway/Data Science - 2/fastapi/model/logistic_empstatus.pkl'
SCALER_PATH = 'D:/Broadway/Data Science - 2/fastapi/model/logistic_scaler.pkl'

def predict_logistic():
    features = ['PerfScoreID', 'Salary', 'PositionID', 'EngagementSurvey', 'EmpSatisfaction',
                'SpecialProjectsCount', 'DaysLateLast30','Absences']
    target = 'Termd'

    X = df[features]
    Y = df[target]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, 
        stratify=Y # for equal distribution of target values
    )

    scaler = StandardScaler()

    X_train_scale = scaler.fit_transform(X_train) # Xi-mean/sd
    X_test_scale = scaler.transform(X_test)

    model = LogisticRegression(
        solver='liblinear', class_weight='balanced', random_state=42
    )
    model.fit(X_train_scale, Y_train)
    model.predict(X_test_scale)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    return model, scaler

# Load model
def load_model_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler