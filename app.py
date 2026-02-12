import os

# Optional upgrade lines
# os.system('pip install --upgrade pip')
# os.system('pip install --upgrade gradio')
os.system("pip install mediapipe")

import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
import mediapipe as mp
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import warnings
warnings.filterwarnings("ignore")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(
    select_largest=False,
    post_process=False,
    device=DEVICE
).to(DEVICE).eval()

model = InceptionResnetV1(
    pretrained="vggface2",
    classify=True,
    num_classes=1,
    device=DEVICE
)

checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# 🟢 MediaPipe Landmark Extractor
def get_mediapipe_landmarks(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            coords = [(lm.x, lm.y, lm.z) for lm in landmarks.landmark]
            return np.array(coords)
        else:
            return None

# 🔍 Main Prediction Function
def predict(input_image: Image.Image):
    face = mtcnn(input_image)
    if face is None:
        raise Exception('No face detected')
    face = face.unsqueeze(0)  # Add batch dimension
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy()
    prev_face = prev_face.astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32) / 255.0

    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # 🔴 Grad-CAM
    target_layers = [model.block8.branch1[-1]]
    use_cuda = torch.cuda.is_available()
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)[0]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    # 🟢 MediaPipe Landmarks Visualization
    landmarks = get_mediapipe_landmarks(input_image)
    if landmarks is not None:
        landmark_image = np.array(input_image).copy()
        h, w, _ = landmark_image.shape
        for (x, y, z) in landmarks:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(landmark_image, (cx, cy), 1, (0, 255, 0), -1)
    else:
        landmark_image = np.array(input_image)

    # 🟨 Convert landmark image to PIL
    landmark_image_pil = Image.fromarray(cv2.cvtColor(landmark_image, cv2.COLOR_BGR2RGB))

    # 🧠 Prediction
    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"
        confidences = {
            'real': 1 - output.item(),
            'fake': output.item()
        }

    return confidences, face_with_mask, landmark_image_pil

# 🖼️ Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Input Image", type="pil"),
    outputs=[
        gr.Label(label="Class"),
        gr.Image(label="Face with Explainability", type="pil"),
        gr.Image(label="Face Landmarks (MediaPipe)", type="pil")
    ],
)

interface.launch(share=True)
