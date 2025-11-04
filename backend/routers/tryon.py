

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from utils.base64_helpers import array_buffer_to_base64
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types
import traceback
import base64

# --- Thư viện mới cho VGG16 ---
import io
from PIL import Image
import numpy as np
# Lưu ý: Cần cài đặt tensorflow và keras-preprocessing
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image
# ------------------------------


load_dotenv()

router = APIRouter()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

# 1. Tải mô hình VGG16 (chỉ 1 lần khi khởi động)
try:
    vgg_model = VGG16(weights="imagenet")
    print("VGG16 model loaded successfully.")
except Exception as e:
    print(f"Error loading VGG16: {e}")
    # Nếu không tải được, vẫn cho phép chạy tiếp mà không dùng VGG16
    vgg_model = None

# 2. Hàm phân tích ảnh bằng VGG16
def analyze_with_vgg16(img_bytes: bytes):
    """Phân tích ảnh bằng VGG16 để lấy nhãn dự đoán"""
    if not vgg_model:
        return [{"label": "VGG16 not loaded", "probability": 0.0}]
    try:
        # Mở ảnh, chuyển sang RGB và resize về 224x224 (yêu cầu của VGG16)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Dự đoán
        preds = vgg_model.predict(img_array, verbose=0)
        # Decode và lấy 3 kết quả hàng đầu
        results = decode_predictions(preds, top=3)[0]

        return [{"label": r[1], "probability": float(r[2])} for r in results]
    except Exception as e:
        print(f"VGG16 analysis error: {e}")
        return [{"label": f"Analysis failed: {str(e)}", "probability": 0.0}]


@router.post("/try-on")
async def try_on(
    place_image: UploadFile = File(...),
    design_type: str = Form(...),
    room_type: str = Form(...),
    style: str = Form(...),
    background_color: str = Form(...),
    foreground_color: str = Form(...),
    instructions: str = Form(""),
):
    try:
        
        MAX_IMAGE_SIZE_MB = 10
        ALLOWED_MIME_TYPES = {
            "image/jpeg",
            "image/png",
            "image/webp",
            "image/heic",
            "image/heif",
        }

        if place_image.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=400, detail=f"Unsupported file type for place_image: {place_image.content_type}"
            )

        place_bytes = await place_image.read()

        size_in_mb_for_place_image = len(place_bytes) / (1024 * 1024)
        if size_in_mb_for_place_image > MAX_IMAGE_SIZE_MB:
            raise HTTPException(status_code=400, detail="Image exceeds 10MB size limit for place_image")
        
        # 3. Phân tích ảnh bằng VGG16 trước khi gửi lên Gemini
        vgg16_analysis = analyze_with_vgg16(place_bytes)
        print("VGG16 Analysis:", vgg16_analysis)
        
        place_b64 = array_buffer_to_base64(place_bytes)

        # 4. Thêm kết quả VGG16 vào prompt
        prompt = f"""
        You are a professional AI interior and exterior designer.
        Your task is to redesign a user's uploaded space.

        ### User Input
        - **Design Type:** {design_type}
        - **Room Type:** {room_type}
        - **Style:** {style}
        - **Background Color Preference:** {background_color}
        - **Foreground Color Preference:** {foreground_color}
        - **Instructions:** {instructions}
        
        ### Image Analysis (VGG16):
        The VGG16 model identified the following top 3 features in the image: {vgg16_analysis}. 
        Use this analysis to better understand the existing objects and context of the room.

        ### Objective:
        1. Apply the chosen design style (e.g., {style}) to the uploaded {room_type}.
        2. Enhance the space visually while respecting the structure of the original layout.
        3. Harmonize background/foreground color preferences subtly in the decor.
        4. Produce a **photo-realistic redesign image** and a **short textual description**.
        5. You don't need change any structure of the room, just the design.
        6. The design should be realistic and practical for the user.
        7. The design should be aligned with the user's preferences and instructions.
        8. Also return the cost and time required for the redesign.
        9. Return the cost of design and the in depth description of the design.
        10. Return all colors of the design in hex format.
        11. Return cost of the design in INR and USD.

        Return:
        - A realistic redesigned image of the space.
        - A short caption describing the redesign, highlighting how it aligns with the selected preferences and suggesting improvements.
        """
        
        print(prompt)

        contents=[
            prompt,
            types.Part.from_bytes(
                data=place_b64,
                mime_type= place_image.content_type,
            )
        ] 
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=contents,
            config=types.GenerateContentConfig(
            response_modalities=['TEXT', 'IMAGE']
            )
        )


        print(response)
        
        image_data = None
        text_response = "No Description available."
        if response.candidates and len(response.candidates) > 0:
            parts = response.candidates[0].content.parts

            if parts:
                print("Number of parts in response:", len(parts))

                for part in parts:
                    if hasattr(part, "inline_data") and part.inline_data:
                        image_data = part.inline_data.data
                        image_mime_type = getattr(part.inline_data, "mime_type", "image/png")
                        print("Image data received, length:", len(image_data))
                        print("MIME type:", image_mime_type)

                    elif hasattr(part, "text") and part.text:
                        text_response = part.text
                        preview = (text_response[:100] + "...") if len(text_response) > 100 else text_response
                        print("Text response received:", preview)
            else:
                print("No parts found in the response candidate.")
        else:
            print("No candidates found in the API response.")

        image_url = None
        if image_data:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            image_url = f"data:{image_mime_type};base64,{image_base64}"
        else:
            image_url = None
    
        return JSONResponse(
            content={
                "image": image_url,
                "text": text_response,
                "vgg16_analysis": vgg16_analysis, # 5. Trả về kết quả VGG16 cho client
            }
        )

    except Exception as e:
        print(f"Error in /api/try-on endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")
