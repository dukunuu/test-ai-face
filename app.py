import gradio as gr
from deepface import DeepFace
from retinaface import RetinaFace
from realesrgan.utils import RealESRGANer
import boto3
import io
import json
from PIL import Image
import numpy as np
import os
import tempfile
from gfpgan import GFPGANer
import urllib.request

# AWS Credentials
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
REGION_NAME = os.getenv("REGION_NAME")

# Initialize S3 Client
s3_client = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=REGION_NAME
)

# Function to download the model if it doesn't exist
def download_model(url, dest_path):
    if not os.path.exists(dest_path):
        print("Downloading model...")
        urllib.request.urlretrieve(url, dest_path)
        print("Model downloaded.")
    else:
        print("Model already exists.")

# Define the model URL and destination path
model_url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
model_dir = "models"
model_path = os.path.join(model_dir, "GFPGANv1.4.pth")
upslampler_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
upsampler_path = os.path.join(model_dir, "realesr-general-x4v3.pth")

# Ensure the models directory exists
os.makedirs(model_dir, exist_ok=True)

# Download the model
download_model(model_url, model_path)
#download_model(upslampler_url, upsampler_path)

#model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
#upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0)
gfpganClient = GFPGANer(model_path=model_path, upscale=2, arch='clean', channel_multiplier=2)

def load_image_from_s3(bucket_name, key, temp_dir):
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        image_bytes = response['Body'].read()
        image = Image.open(io.BytesIO(image_bytes))
        student_code = key.split('/')[-1].split('.')[0]
        image_path = os.path.join(temp_dir, student_code + '.jpg')
        image.save(image_path)
        return image_path
    except Exception as e:
        print(f"Error loading {key} from S3: {str(e)}")
        return None

def preprocess_image(image):
    return np.array(image)

def upscale_image(image):
    # GFP-GAN expects PIL images
    print("Upscaling image...")
    _, result, _ = gfpganClient.enhance(image, has_aligned=True)
    upscaled_image = result[0]
    return np.array(upscaled_image)

def detect_and_recognize_faces(image_file, course_id):
    try:
        student_images_key = f"courses/{course_id}/student_images.json"
        response = s3_client.get_object(Bucket=AWS_BUCKET_NAME, Key=student_images_key)
        student_image_keys = json.loads(response['Body'].read())
        print(f"Student image keys: {student_image_keys}")

        uploaded_image_pil = Image.open(image_file)
        uploaded_image_np = preprocess_image(uploaded_image_pil)

        detected_faces = RetinaFace.extract_faces(img_path=uploaded_image_np, align=True)
        print(f"Detected {len(detected_faces)} faces in the uploaded image.")

        if not detected_faces:
            return {"error": "No faces detected in the uploaded image."}

        face_results = []

        with tempfile.TemporaryDirectory() as temp_dir:
            student_image_paths = []
            for student_key in student_image_keys:
                image_path = load_image_from_s3(AWS_BUCKET_NAME, student_key, temp_dir)
                if image_path:
                    student_image_paths.append(image_path)

            for face in detected_faces:
                upscaled_face = upscale_image(face)

                df_results = DeepFace.find(
                    img_path=upscaled_face,
                    db_path=temp_dir,
                    model_name="Facenet512",
                    enforce_detection=True,
                    align=True,
                    detector_backend='retinaface',
                    silent=True,
                    threshold=0.4
                )

                matched_faces = df_results[0]
                if not matched_faces.empty:
                    most_similar = matched_faces.iloc[0]
                    student_code = most_similar['identity'].split('/')[-1].split('.')[0]
                    distance = most_similar['distance']
                    face_results.append({
                        "student_code": student_code,
                        "distance": distance
                    })

    except Exception as e:
        return {"error": str(e)}

    unique_student_codes = set([result["student_code"] for result in face_results])
    return {"results": list(unique_student_codes)}

iface = gr.Interface(
    fn=detect_and_recognize_faces,
    inputs=[
        gr.File(label="Image File"),
        gr.Textbox(label="Course ID")
    ],
    outputs=gr.JSON(label="Detection Results"),
    title="Face Detection & Recognition",
    description="Upload an image file and course ID to detect and recognize faces using DeepFace and RetinaFace."
)

iface.launch(debug=True)

