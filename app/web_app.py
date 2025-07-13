import streamlit as st
import os
import tempfile
from inference import predict_image
from PIL import Image

# Konfiguration der Streamlit-Webseite
st.set_page_config(page_title="Zahlenerkennung", layout="centered")

# Überschrift und Beschreibung
st.title("Zahlenerkennung mit CNN")
st.write("Lade ein handgeschriebenes Bild hoch – das Modell erkennt die Zahl und den Verfasser.")

# Dateiupload über die Web-Oberfläche
uploaded_file = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Bild aus dem Upload laden und in RGB konvertieren (sichert Kompatibilität)
    image = Image.open(uploaded_file).convert("RGB")

    # Vorschau des Bildes anzeigen
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Temporäre Datei speichern, da predict_image mit einem Dateipfad arbeitet
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Modell-Inferenz auf dem hochgeladenen Bild ausführen
    with st.spinner("Analysiere das Bild..."):
        digit, person, digit_conf, person_conf = predict_image(temp_path)

    # Ergebnisse visuell ausgeben
    st.success(f" **Zahl:** {digit} ({digit_conf:.2%} sicher)")
    st.info(f" **Person:** {person} ({person_conf:.2%} sicher)")

    # Aufräumen: Temporäre Bilddatei löschen
    os.remove(temp_path)

# Zusatzfunktion: Ermöglicht den Download des trainierten Modells
st.subheader(" Modell herunterladen")

model_path = "model/cnn_model.h5"

if os.path.exists(model_path):
    # Ermöglicht es, das Modell als .h5-Datei herunterzuladen
    with open(model_path, "rb") as model_file:
        st.download_button(
            label=" CNN-Modell herunterladen (.h5)",
            data=model_file,
            file_name="cnn_model.h5",
            mime="application/octet-stream"
        )
else:
    # Fehlermeldung, falls die Datei fehlt
    st.error("❌ Modell-Datei nicht gefunden unter 'model/cnn_model.h5'")