from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Mimari stiller
MIMARI_STILLER = [
    "barok", "brütalizm", "deco", "dekonstrüktivizm",
    "gotik", "modern", "neoklasik", "nouveau",
    "post-modern", "rokoko", "rönesans"
]

# Model yükleme (ResNet18 + Dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=False)
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, len(MIMARI_STILLER))
)
model.load_state_dict(torch.load("best_architecture.pt", map_location=device))
model = model.to(device)
model.eval()

# Görüntü preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0).to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Dosya yüklenmedi'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'})
    
    try:
        image_bytes = file.read()
        input_tensor = preprocess_image(image_bytes)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs[0], dim=0).cpu().numpy()

        results = {
            "labels": MIMARI_STILLER,
            "values": probs.tolist()
        }

        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
