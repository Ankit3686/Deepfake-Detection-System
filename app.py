import os
import gradio as gr
import torch
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from fpdf import FPDF
import uuid
import warnings

warnings.filterwarnings("ignore")

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load MTCNN
mtcnn = MTCNN(select_largest=False, post_process=False, device=DEVICE).to(DEVICE).eval()

# Load Model
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


# Main Prediction
def predict(input_image: Image.Image):

    face = mtcnn(input_image)

    if face is None:
        blank = Image.new("RGB", input_image.size if input_image else (256, 256), color=(255, 255, 255))
        return "No face detected", {"real": 0.0, "fake": 0.0}, blank, blank, blank, None

    face = face.unsqueeze(0)
    face = F.interpolate(face, size=(256, 256), mode='bilinear', align_corners=False)

    prev_face = face.squeeze(0).permute(1, 2, 0).cpu().detach().int().numpy().astype('uint8')

    face = face.to(DEVICE)
    face = face.to(torch.float32) / 255.0
    face_image_to_plot = face.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()

    # Grad-CAM
    target_layers = [model.block8.branch1[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]

    grayscale_cam = cam(input_tensor=face, targets=targets, eigen_smooth=True)[0]
    visualization = show_cam_on_image(face_image_to_plot, grayscale_cam, use_rgb=True)
    face_with_mask = cv2.addWeighted(prev_face, 1, visualization, 0.5, 0)

    # Convert to PIL
    face_with_mask_pil = Image.fromarray(face_with_mask)

    cropped_face_np = face.squeeze(0).permute(1, 2, 0).cpu().numpy()
    cropped_face_pil = Image.fromarray((cropped_face_np * 255).astype(np.uint8))

    # Prediction
    with torch.no_grad():
        output = torch.sigmoid(model(face).squeeze(0))
        prediction = "real" if output.item() < 0.5 else "fake"
        confidences = {
            "real": round(1 - output.item(), 4),
            "fake": round(output.item(), 4)
        }

    # PDF
    pdf_filename = f"/tmp/prediction_report_{uuid.uuid4().hex}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Fake Face Detection Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction Verdict: {prediction}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence - Real: {confidences['real']}, Fake: {confidences['fake']}", ln=True)

    face_with_mask_path = "/tmp/mask.jpg"
    cropped_face_path = "/tmp/cropped.jpg"

    face_with_mask_pil.save(face_with_mask_path)
    cropped_face_pil.save(cropped_face_path)

    for img_path, label in [
        (face_with_mask_path, "Grad-CAM Visualization"),
        (cropped_face_path, "Cropped Face")
    ]:
        pdf.ln(5)
        pdf.cell(200, 10, txt=label, ln=True)
        pdf.image(img_path, x=10, w=100)

    pdf.output(pdf_filename)

    return prediction, confidences, face_with_mask_pil, face_with_mask_pil, cropped_face_pil, pdf_filename


# Clear
def clear_all():
    return None, "", "", {}, None, None, None, None


# UI
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Fake Face Detector")

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
            explain = gr.Image(label="Explainability", type="pil")
            landmarks_img = gr.Image(label="Face Landmarks (Disabled)", type="pil")
            cropped = gr.Image(label="Cropped Face", type="pil")

    def predict_with_status(input_image):
        yield gr.update(visible=True, value="⏳ Processing..."), None, None, None, None, None, None
        result = predict(input_image)
        yield gr.update(visible=True, value=f"✅ Done: {result[0]}"), *result

    run_btn.click(
        fn=predict_with_status,
        inputs=img_input,
        outputs=[status_box, verdict, scores, explain, landmarks_img, cropped, file_download]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[img_input, status_box, verdict, scores, explain, landmarks_img, cropped, file_download]
    )

demo.launch(share=True)
