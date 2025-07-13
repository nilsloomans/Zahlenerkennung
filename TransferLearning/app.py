import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from image_utils import preprocess_image

# Exakt deine Modellstruktur
class MultiTaskResNet(nn.Module):
    def __init__(self):
        super(MultiTaskResNet, self).__init__()
        base = models.resnet18(pretrained=False)  # pretrained=False, da du Gewichtelädst
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head_digit = nn.Linear(base.fc.in_features, 10)
        self.head_person = nn.Linear(base.fc.in_features, 3)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        return self.head_digit(x), self.head_person(x)

# Modell laden
@st.cache_resource
def load_model():
    model = MultiTaskResNet()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

# Streamlit UI
st.title("Zahlenerkennung mit einem CNN (Transfer Learning)")
st.write("Lade ein Bild hoch, um Person und Ziffer vorherzusagen.")

# 🟢 Modell-Download immer sichtbar
with open("model.pth", "rb") as f:
    model_bytes = f.read()

st.download_button(
    label="📥 Modell herunterladen",
    data=model_bytes,
    file_name="model.pth",
    mime="application/octet-stream"
)

uploaded_file = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    model = load_model()
    img_array = preprocess_image(uploaded_file)

    if img_array is not None:
        input_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]

        with torch.no_grad():
            digit_logits, person_logits = model(input_tensor)
            predicted_digit = torch.argmax(digit_logits, dim=1).item()
            predicted_person = torch.argmax(person_logits, dim=1).item()

        person_labels = {0: "Tim", 1: "Thanadon", 2: "Nils"}  # ← deine Zuordnung hier anpassen
        predicted_person_name = person_labels.get(predicted_person, f"Unbekannt ({predicted_person})")

        st.markdown("### Vorhersage")
        st.write(f"**Zahl:** {predicted_digit}")
        st.write(f"**Person:** {predicted_person_name}")