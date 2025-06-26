from app.inference import predict_image

digit, person = predict_image("data/unknown_samples/testbild.jpg")
print(f"Zahl erkannt: {digit}")
print(f"Person erkannt: {person}")