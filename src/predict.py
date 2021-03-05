import os
import pandas as pd
import numpy as np
import joblib


TEST_DATA = os.environ.get('TEST_DATA')
MODEL_TYPE = os.environ.get('MODEL')
model_path = "models/"


def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["id"].values
    predictions = None
    
    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(TEST_DATA)
        encoders = joblib.load(os.path.join(model_path, f"{MODEL_TYPE}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join(model_path, f"{MODEL_TYPE}_{FOLD}_columns.pkl"))
        for c in cols:
            lbl = encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
        
        clf = joblib.load(os.path.join(model_path, f"{MODEL_TYPE}_{FOLD}.pkl"))
        
        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]
        
        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5
    
    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    return sub


if __name__ == '__main__':
    submission = predict()
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"{model_path}submission_{MODEL_TYPE}.csv", index=False)
    