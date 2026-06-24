from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib, os

app = Flask(__name__)

obj              = joblib.load("modelo_co2.pkl")
pipeline         = obj["pipeline"]
ord_enc          = obj["ordinal_encoder"]
cat_cols         = obj["encoder_cat_cols"]
es_pca           = obj["es_pca"]
features_sel     = obj["features_sel"]
features_all     = obj["features_all"]

FUEL_LABELS = {"X": "Gasolina regular", "Z": "Gasolina premium",
               "E": "Etanol (E85)", "D": "Diesel", "N": "Gas natural"}

def get_ctx():
    makes    = sorted(ord_enc.categories_[cat_cols.index("Make")].tolist())
    classes  = sorted(ord_enc.categories_[cat_cols.index("Vehicle Class")].tolist())
    trans    = sorted(ord_enc.categories_[cat_cols.index("Transmission")].tolist())
    return dict(fuel_options=list(FUEL_LABELS.items()),
                makes=makes, classes=classes, transmissions=trans)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **get_ctx())

@app.route("/predict", methods=["POST"])
def predict():
    ctx = get_ctx()
    try:
        raw = {
            "Make"                             : request.form["Make"],
            "Vehicle Class"                    : request.form["Vehicle_Class"],
            "Engine Size(L)"                   : float(request.form["Engine_Size"]),
            "Cylinders"                        : int(request.form["Cylinders"]),
            "Transmission"                     : request.form["Transmission"],
            "Fuel Type"                        : request.form["Fuel_Type"],
            "Fuel Consumption City (L/100 km)" : float(request.form["Fuel_City"]),
            "Fuel Consumption Hwy (L/100 km)"  : float(request.form["Fuel_Hwy"]),
            "Fuel Consumption Comb (L/100 km)" : float(request.form["Fuel_Comb"]),
            "Fuel Consumption Comb (mpg)"      : int(request.form["Fuel_mpg"])
        }
        df_in = pd.DataFrame([raw])
        df_in[cat_cols] = ord_enc.transform(df_in[cat_cols])
        X_in = df_in[features_sel] if not es_pca else df_in[features_all]
        pred = round(float(pipeline.predict(X_in)[0]), 1)
        nivel = ("Bajo (<150 g/km)"   if pred < 150 else
                 "Moderado (150-250)" if pred < 250 else
                 "Alto (250-350)"     if pred < 350 else "Muy alto (>350)")
        return render_template("index.html", prediccion=pred, nivel=nivel, datos=raw, **ctx)
    except Exception as exc:
        return render_template("index.html", error=str(exc), **ctx)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
