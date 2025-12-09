from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Path
from fastapi.responses import JSONResponse
from utils.db_client import supabase
import base64
import traceback

router = APIRouter()

@router.post("/furniture/upload")
async def upload_furniture(
    session_id: str = Form(...),
    name: str = Form(""),
    furniture_image: UploadFile = File(...)
):
    try:
        allowed_types = ["image/png", "image/jpeg", "image/webp"]
        if furniture_image.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Unsupported image type")

        bytes_data = await furniture_image.read()
        b64 = base64.b64encode(bytes_data).decode("utf-8")

        data = {
            "session_id": session_id,
            "description": name,
            "image_base64": b64
        }

        result = supabase.table("furnitures").insert(data).execute()

        return JSONResponse({
            "status": "success",
            "message": "Furniture uploaded",
            "id": result.data[0]["id"]
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Upload failed")

@router.get("/furniture/all")
async def get_all_furniture(session_id: str = None):
    try:
        query = supabase.table("furnitures").select("id, session_id, created_at, image_base64, description").order("created_at", desc=True)
        if session_id:
            query = query.eq("session_id", session_id)
        result = query.execute()

        items = result.data or []

        furnitures = [
            {
                "id": item["id"],
                "name": item.get("description") or "Furniture",
                "created_at": item["created_at"],
                "image": f"data:image/png;base64,{item['image_base64']}"
            } for item in items
        ]
        return {"furnitures": furnitures}
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to load furniture library")
    
@router.delete("/furniture/{furniture_id}")
async def delete_furniture(furniture_id: str = Path(..., description="ID of the furniture")):
    try:
        result = supabase.table("furnitures").delete().eq("id", furniture_id).execute()
        if not result.data:  # empty list means no matching row
            raise HTTPException(status_code=404, detail="Furniture not found")
        return {"status": "success", "message": "Furniture deleted"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
