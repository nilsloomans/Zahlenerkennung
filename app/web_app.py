import streamlit as st
import os
import tempfile
from inference import predict_image
from PIL import Image

st.set_page_config(page_title="Zahlenerkennung", layout="centered")

st.title("🔢 Zahlenerkennung mit CNN")
st.write("Lade ein handgeschriebenes Bild hoch – das Modell erkennt die Zahl und den Verfasser.")

# Bild hochladen
uploaded_file = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Bild anzeigen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Temporär speichern für predict_image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Vorhersage aufrufen
    with st.spinner("🔍 Analysiere das Bild..."):
        digit, person, digit_conf, person_conf = predict_image(temp_path)

    # Ergebnisse anzeigen
    st.success(f"📌 **Zahl:** {digit} ({digit_conf:.2%} sicher)")
    st.info(f"✍️ **Person:** {person} ({person_conf:.2%} sicher)")

    # Temporäre Datei löschen
    os.remove(temp_path)

# 📥 Modell-Download unabhängig vom Bild
st.subheader("📦 Modell herunterladen")

model_path = "model/cnn_model.h5"

if os.path.exists(model_path):
    with open(model_path, "rb") as model_file:
        st.download_button(
            label="📥 CNN-Modell herunterladen (.h5)",
            data=model_file,
            file_name="cnn_model.h5",
            mime="application/octet-stream"
        )
else:
    st.error("❌ Modell-Datei nicht gefunden unter 'model/cnn_model.h5'")