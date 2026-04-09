from app.ml.generate_pdf_file import PatientDataFile

def test_generate_pdf_file():
    patient_data = {
        'Age': 45,
        'Cholesterol': 210,
        'Max HR': 150
    }
    probability = 75.5
    img_base64 = None

    pdf_bytes = PatientDataFile.generate_pdf(patient_data, probability, img_base64)

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")