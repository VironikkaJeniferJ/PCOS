from flask import Flask, render_template, request
import numpy as np 
import joblib 
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
with open('rf.pkl', 'rb') as file:
    model = joblib.load(file)
@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        follicle_r = float(request.form["follicle_r"])
        follicle_l = float(request.form["follicle_l"])
        skin = float(request.form["skin"])
        hair_g= float(request.form["hair_g"])
        weight_g=float(request.form["weight_g"])
        cycle=float(request.form["cycle"])
        fast_food=float(request.form["fast_food"])
        pimples=float(request.form["pimples"])
        hair_loss=float(request.form["hair_loss"])
        weight_kg=float(request.form["weight_kg"])
        bmi=float(request.form["bmi"])
        waist=float(request.form["waist"])
        hip=float(request.form["hip"])
        avg_f_size_l=float(request.form["avg_f_size_l"])
        endometrium=float(request.form["endometrium"])
        epsilon = 1e-10
        amh_input = float(request.form["amh_input"]) + epsilon
        log_amh = np.log10(amh_input)
        #features = [float(request.form.get(f'feature_{i}')) for i in range(16)]
        X_train = np.load('X_train.npy')
        input_data = [follicle_r, follicle_l, skin, hair_g, weight_g, cycle, fast_food, pimples, hair_loss, log_amh, weight_kg, bmi, waist, hip, avg_f_size_l, endometrium]
        #prediction = model.predict([input_data])
        features = np.array(input_data).reshape(1, -1) 
        scaler= StandardScaler()
        scaler.fit(X_train)
        standardized_new_data = scaler.transform(features)
        prediction = model.predict(standardized_new_data)[0]
        if prediction == 1:
            return render_template("result.html")
        else:
            return render_template("resultno.html")
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)