# Zahlenerkennung

Projektbeschreibung
Dieses Projekt umfasst zwei Ansätze zur Erkennung handgeschriebener Ziffern (0–9) und der schreibenden Person (Tim, Thanadon oder Nils) anhand von Bilddaten:

Eigenes CNN-Modell: Von Grund auf entwickeltes Convolutional Neural Network mit zwei Ausgängen (Zahl und Person).

Transfer Learning Modell: Auf ResNet18 basierendes Multi-Task-Modell mit zwei Klassifikationsköpfen.

Ziel ist es, aus einem hochgeladenen Bild die enthaltene Ziffer und den Verfasser zu ermitteln.

Voraussetzungen
Für beide Modelle
Python ≥ 3.10

Virtuelle Umgebung empfohlen (venv oder conda)

CNN-Modell (TensorFlow-Version)
pip install tensorflow numpy matplotlib scikit-learn streamlit pillow

Transfer Learning Modell (PyTorch-Version)
pip install torch torchvision numpy pillow streamlit

Web-Anwendung starten
Eigenes CNN-Modell
Stelle sicher, dass das Modell cnn_model.h5 unter model/ liegt.

Starte die App:
streamlit run app/web_app.py
oder:
python -m streamlit run app/web_app.py
oder:
Direkt aus dem app Ordner -> Dann kann das app/ weggelassen werden

Transfer Learning Modell
Stelle sicher, dass model.pth im Projektverzeichnis liegt.

Starte die App:
streamlit run app.py
oder:
python -m streamlit run TransferLearning/app.py
oder:
Direkt aus dem TransferLearning Ordner -> Dann kann das TransferLearning/ weggelassen werden
