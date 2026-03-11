from fpdf import FPDF
import base64
import tempfile
import os

class PatientDataFile:

    @staticmethod
    def generate_pdf(patient_data : dict, probability : float, img_base64):
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", 'B', 16)

        pdf.cell(0, 10, txt="Heart Disease Risk Analysis Report", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="Patient Data:", ln=True)
        pdf.set_font("Arial", '', 11)

        for key, value in patient_data.items():
            pdf.cell(0, 8, txt=f"- {key}: {value}", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="Analysis Result:", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, txt=f"Probability of disease: {probability:.1f}%", ln=True)

        if probability >= 70:
            recommendation = "High risk! It is recommended to urgently consult a cardiologist."
        elif probability >= 45:
            recommendation = "Increased risk. Schedule a routine visit to your doctor."
        else:
            recommendation = "Low risk! Continue to lead a healthy lifestyle."

        pdf.multi_cell(0, 8, txt=f"Recommendation: {recommendation}")
        pdf.ln(10)

        if img_base64:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, txt="Model Explanation (SHAP):", ln=True)

            img_data = base64.b64decode(img_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_file.write(img_data)
                tmp_file_path = tmp_file.name

            pdf.image(tmp_file_path, w=170)

            os.remove(tmp_file_path)

        return bytes(pdf.output())

