from flask import Flask, render_template, request
import pandas as pd
import pickle
#Create 
app = Flask(__name__)

# Load the trained model and column transformer
def prediction_input_data(input_df):
    model = pickle.load(open("model.pkl", "rb"))
    ct=pickle.load(open("columntransformer.pkl","rb"))
    transformer=ct.fit_transform(input_df)
    ans = round(model.predict(transformer)[0])
    ans= str(ans)
    ans=ans[0:2]+","+ans[2:]+"$"
    return (ans)

@app.route("/")
def display_form():
    return render_template("home.html")
#root
@app.route("/predict", methods=["POST"])
def get_input_data():
    input_data = [
        (request.form["JobTitle"]),
        int(request.form["Age"]),
        (request.form["Gender"]),
        int(request.form["PerfEval"]),
        request.form["Education"],
        request.form["Dept"],
        int(request.form["Seniority"])
    ]

    input_df = pd.DataFrame(data=[input_data], columns=['JobTitle','Age', 'Gender', 'PerfEval', 'Education', 'Dept', 'Seniority'])

    ans = prediction_input_data(input_df)
    return render_template("display.html", data=ans)

if __name__ == "__main__":
    app.run(debug=True)
