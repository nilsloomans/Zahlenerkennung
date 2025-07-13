#  Zahlenerkennung

##  Projektbeschreibung

Dieses Projekt umfasst zwei Ansätze zur Erkennung handgeschriebener Ziffern (0–9) und der schreibenden Person (Tim, Thanadon oder Nils) anhand von Bilddaten:

- **Eigenes CNN-Modell**  
  Von Grund auf entwickeltes Convolutional Neural Network mit zwei Ausgängen (Zahl und Person).

- **Transfer Learning Modell**  
  Auf ResNet18 basierendes Multi-Task-Modell mit zwei Klassifikationsköpfen.

Ziel ist es, aus einem hochgeladenen Bild die enthaltene Ziffer und den Verfasser zu ermitteln.

---

##  Voraussetzungen

### Für beide Modelle

- Python 3.11.9 empfohlen (mind. Python ≥ 3.10, damit keine Probleme auftreten)
- Virtuelle Umgebung empfohlen (venv oder conda)

### CNN-Modell (TensorFlow-Version)

pip install tensorflow numpy matplotlib scikit-learn streamlit pillow

### Transfer Learning Modell (PyTorch-Version)

pip install torch torchvision numpy pillow streamlit

---

##  Web-Anwendung starten

### Eigenes CNN-Modell

1. Stelle sicher, dass das Modell cnn_model.h5 unter model/ liegt.  
2. Starte die App:

streamlit run app/web_app.py

Alternativen:

python -m streamlit run app/web_app.py

Oder direkt aus dem app-Ordner heraus:

streamlit run web_app.py

---

### Transfer Learning Modell

1. Stelle sicher, dass model.pth im Projektverzeichnis liegt.  
2. Starte die App:

streamlit run TransferLearning/app.py

Alternativen:

python -m streamlit run TransferLearning/app.py

Oder direkt aus dem TransferLearning-Ordner heraus:

streamlit run app.py
