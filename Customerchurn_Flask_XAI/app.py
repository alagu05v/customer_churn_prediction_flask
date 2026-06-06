from flask import Flask, render_template, request, send_file
import pandas as pd
import joblib
import shap
import numpy as np
import os

# Matplotlib backend fix for Flask
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PDF generation (Professional format)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem

app = Flask(__name__)

# Load model and columns
model = joblib.load("model/xgb_model.pkl")
columns = joblib.load("model/columns.pkl")


# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        gender = request.form["gender"]
        senior = int(request.form["senior"])
        partner = request.form["partner"]
        dependents = request.form["dependents"]
        tenure = float(request.form["tenure"])
        phone = request.form["phone"]
        internet = request.form["internet"]
        contract = request.form["contract"]
        payment = request.form["payment"]
        monthly = float(request.form["monthly"])
        total = float(request.form["total"])

        # Create dataframe with all columns
        input_data = pd.DataFrame(0, index=[0], columns=columns)

        # Assign numeric values
        input_data["SeniorCitizen"] = senior
        input_data["tenure"] = tenure
        input_data["MonthlyCharges"] = monthly
        input_data["TotalCharges"] = total

        # One-hot encoding manually
        def set_column(col):
            if col in input_data.columns:
                input_data[col] = 1

        set_column(f"gender_{gender}")
        set_column(f"Partner_{partner}")
        set_column(f"Dependents_{dependents}")
        set_column(f"PhoneService_{phone}")
        set_column(f"InternetService_{internet}")
        set_column(f"Contract_{contract}")
        set_column(f"PaymentMethod_{payment}")

        # Predict probability
        prob = model.predict_proba(input_data)[:, 1][0]
        threshold = 0.35
        prediction = "Churn" if prob >= threshold else "Not Churn"

        # SHAP Explanation
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(input_data)

        # Save SHAP plot
        plt.figure(figsize=(8,5))
        shap.plots.bar(shap_values[0], max_display=8, show=False)
        plt.tight_layout()
        plt.savefig("static/shap_plot.png", bbox_inches="tight", dpi=300)
        plt.close()

        # Top 5 Features
        feature_importance = pd.DataFrame({
            "Feature": input_data.columns,
            "SHAP_Value": shap_values.values[0]
        })

        feature_importance["Impact"] = feature_importance["SHAP_Value"].apply(
            lambda x: "Increases Churn" if x > 0 else "Decreases Churn"
        )

        top5 = feature_importance.reindex(
            feature_importance["SHAP_Value"].abs().sort_values(ascending=False).index
        ).head(5)

        top5_list = top5.to_dict(orient="records")

        # Store for PDF
        app.config["prediction"] = prediction
        app.config["probability"] = round(prob * 100, 2)
        app.config["threshold"] = threshold
        app.config["top5"] = top5_list

        return render_template(
            "result.html",
            prediction=prediction,
            probability=round(prob * 100, 2),
            threshold=threshold,
            top5=top5_list
        )

    except Exception as e:
        return f"Error occurred: {str(e)}"


# ---------------- DOWNLOAD PDF ----------------
@app.route("/download_report")
def download_report():

    pdf_path = "static/churn_report.pdf"
    doc = SimpleDocTemplate(pdf_path)
    elements = []

    styles = getSampleStyleSheet()

    # Title
    elements.append(Paragraph("<b>Customer Churn Prediction Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Prediction details
    elements.append(Paragraph(f"<b>Prediction:</b> {app.config.get('prediction')}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Churn Probability:</b> {app.config.get('probability')}%", styles["Normal"]))
    elements.append(Paragraph(f"<b>Threshold Used:</b> {app.config.get('threshold')}", styles["Normal"]))
    elements.append(Spacer(1, 0.3 * inch))

    # Top 5 Reasons
    elements.append(Paragraph("<b>Top 5 SHAP Contributing Features:</b>", styles["Heading3"]))
    elements.append(Spacer(1, 0.2 * inch))

    reasons = []
    for r in app.config.get("top5", []):
        reasons.append(
            ListItem(
                Paragraph(
                    f"{r['Feature']} ({r['Impact']}) - SHAP Value: {round(r['SHAP_Value'], 4)}",
                    styles["Normal"]
                )
            )
        )

    elements.append(ListFlowable(reasons, bulletType="bullet"))
    elements.append(Spacer(1, 0.5 * inch))

    # Add SHAP image
    if os.path.exists("static/shap_plot.png"):
        elements.append(Paragraph("<b>SHAP Feature Importance Plot:</b>", styles["Heading3"]))
        elements.append(Spacer(1, 0.2 * inch))
        elements.append(Image("static/shap_plot.png", width=5 * inch, height=3 * inch))

    doc.build(elements)

    return send_file(pdf_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)