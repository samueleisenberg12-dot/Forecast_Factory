from fastapi import FastAPI, File, UploadFile
from typing_extensions import Annotated
import uvicorn
import io
import traceback
from utils import *

# create FastAPI app
app = FastAPI()

# global variable for data set
uploaded_data_set = None

@app.get("/")
async def root():
    return {"message": "Welcome to the Forecast Factory"}

@app.post("/upload_data_csv/")
async def create_upload_file(file: UploadFile):
    global uploaded_data_set
    if not file.filename.lower().endswith('.csv'):
        return {"Upload Error": "Invalid file type"}
    else:
        contents = await file.read() 
        uploaded_data_set = pd.read_csv(io.BytesIO(contents))
        return {"Upload Success": f"{file.filename}"}
    
@app.get("/find_2_day_ahead_forecast/")
async def get_forecast(demand_column_str: str, day_column_str: str):
    global uploaded_data_set
    if uploaded_data_set is None:
        return {"Solver Error": "No active data set, please upload a data set first."}
    elif demand_column_str not in uploaded_data_set.columns or day_column_str not in uploaded_data_set.columns:
        return {"Invalid demand or day column name"}
    else:

        df_cleaned, ok = clean_df(uploaded_data_set, demand_column_str, day_column_str)
        if not ok:
            return {"Invalid data: Dates are not spaced 1 day apart"}
        
        data_train, data_validate, data_test = split_data(0.7, 0.1, df_cleaned)

        data_train, data_validate, data_test, scaler = separate_and_normalize(
            data_train, data_validate, data_test
        )

        Xtrain, ytrain, ttrain = organize_data(data_train['scaled'])
        Xvalid, yvalid, tvalid = organize_data(data_validate['scaled'])
        Xtest, ytest, ttest = organize_data(data_test['scaled'])

        Xtrain_day = append_day(Xtrain,ttrain)
        Xvalid_day = append_day(Xvalid, tvalid)
        Xtest_day = append_day(Xtest, ttest)

        linreg_model, linreg_r2 = linreg_eval(Xtrain, Xtrain_day, ytrain, Xvalid, Xvalid_day, yvalid)

        lstm_model, lstm_r2 = lstm_eval(Xtrain_day, ytrain, Xvalid_day, Xvalid, yvalid)

        best_model_name, best_model_r2 = best_perf(linreg_r2, lstm_r2)

        if best_model_r2 == lstm_r2:
            best_model = lstm_model
        elif best_model_r2 == linreg_r2:
            best_model = linreg_model

        forecast_2day = forecast(best_model, df_cleaned, demand_column_str, 2, scaler)

        return {"Model": best_model_name,
                "Performance": best_model_r2,
                "Forecast": forecast_2day
                }
    
if __name__ == "__main__":
    print("Server is running at http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080)