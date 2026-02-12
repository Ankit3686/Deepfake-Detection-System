import os
# Optional upgrade lines
# os.system('pip install --upgrade pip')
# os.system('pip install --upgrade gradio')
os.system("pip install mediapipe")
os.system("pip install fpdf")

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
from fpdf import FPDF
import uuid
import warnings
import time
warnings.filterwarnings("ignore")

# 💻 Set device
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 🧠 Load MTCNN for face detection
mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()

# 🧠 Load InceptionResnetV1 model
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
        return "No face detected", {"real": 0.0, "fake": 0.0}, input_image, input_image, input_image, None

    face = face.unsqueeze(0)
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    # Save original face before normalization
    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy().astype('uint8')

    # Prepare face for model
    face = face.to(DEVICE)
    face = face.to(torch.float32) / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # 🔴 Grad-CAM Visualization
    target_layers = [model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)[0]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    # 🟢 MediaPipe Landmarks Visualization
    landmarks = get_mediapipe_landmarks(input_image)
    landmark_image = np.array(input_image).copy()
    if landmarks is not None:
        h, w, _ = landmark_image.shape
        for (x, y, z) in landmarks:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(landmark_image, (cx, cy), 1, (0, 255, 0), -1)

    # 🟨 Convert to PIL images
    landmark_image_pil = Image.fromarray(cv2.cvtColor(landmark_image, cv2.COLOR_BGR2RGB))
    face_with_mask_pil = Image.fromarray(face_with_mask)
    cropped_face_np = face.squeeze(0).permute(1, 2, 0).cpu().numpy()
    cropped_face_pil = Image.fromarray((cropped_face_np * 255).astype(np.uint8))

    # 🧠 Make prediction
    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"
        confidences = {
            "real": round(1 - output.item(), 4),
            "fake": round(output.item(), 4)
        }

    # 📄 Generate PDF Report
    pdf_filename = f"/tmp/prediction_report_{uuid.uuid4().hex}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Fake Face Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction Verdict: {prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence - Real: {confidences['real']}, Fake: {confidences['fake']}", ln=True)

    # Save and add images
    face_with_mask_path = "/tmp/mask.jpg"
    landmarks_path = "/tmp/landmarks.jpg"
    cropped_face_path = "/tmp/cropped.jpg"
    face_with_mask_pil.save(face_with_mask_path)
    landmark_image_pil.save(landmarks_path)
    cropped_face_pil.save(cropped_face_path)

    for img_path, label in [(face_with_mask_path, "Grad-CAM Visualization"),
                            (landmarks_path, "Face Landmarks"),
                            (cropped_face_path, "Cropped Face")]:
        pdf.ln(5)
        pdf.cell(200, 10, txt=label, ln=True)
        pdf.image(img_path, x=10, w=100)

    pdf.output(pdf_filename)

    return prediction, confidences, face_with_mask_pil, landmark_image_pil, cropped_face_pil, pdf_filename

 # 🧹 Clear all UI components
def clear_all():
    return None, "", "", {}, None, None, None, None

# 🧠 Enhanced Gradio Blocks UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Fake Face Detector")
    gr.Markdown("Upload an image and click **Run Detection** to find out if it's real or fake.")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="Input Image", type="pil")
            file_download = gr.File(label="📄 Download Report PDF")

            run_btn = gr.Button("▶️ Run Detection")
            clear_btn = gr.Button("🔁 Clear") 
            status_box = gr.Textbox(label="Status", visible=False)

        with gr.Column():
            verdict = gr.Textbox(label="Prediction Verdict")
            scores = gr.Label(label="Confidence Scores")
            explain = gr.Image(label="Face with Explainability", type="pil")
            landmarks = gr.Image(label="Face Landmarks (MediaPipe)", type="pil")
            cropped = gr.Image(label="Cropped Face", type="pil")

    # Function that adds status updates
    def predict_with_status(input_image):
        yield gr.update(visible=True, value="⏳ Processing..."), None, None, None, None, None, None
        result = predict(input_image)
        yield gr.update(visible=True, value=f"✅ Done. Prediction: {result[0]}"), *result


    # Connect button to prediction function
    run_btn.click(
        fn=predict_with_status,
        inputs=img_input,
        outputs=[status_box, verdict, scores, explain, landmarks, cropped, file_download]
    )

    # Connect clear button to clear_all function
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[img_input, status_box, verdict, scores, explain, landmarks, cropped, file_download]
    )


demo.launch(share=True)
