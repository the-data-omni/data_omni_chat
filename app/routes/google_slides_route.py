"""
`This file is used to connect to google slides and generate a google slide from analysis in the chat
"""
import io
from typing import List, Optional

import google.oauth2.credentials
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.security import OAuth2PasswordBearer
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload
from pydantic import BaseModel

router = APIRouter()

# This will extract the token from the Authorization header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class SlideData(BaseModel):
    """Model for slide Data"""
    title: str
    image: Optional[str] = None
    summary: str

class PresentationData(BaseModel):
    """Model for presentation Data"""
    title: str
    slides: List[SlideData]

def get_user_credentials(token: str = Depends(oauth2_scheme)):
    """Authenticates the user via the token from the frontend."""
    return google.oauth2.credentials.Credentials(token)


@router.post("/upload/convert-pptx-to-slides")
async def convert_pptx_to_slides(
    
    creds: google.oauth2.credentials.Credentials = Depends(get_user_credentials),
    pptx_file: UploadFile = File(...)
):
    """converts powerpoint to google slides because we cannot generate slides directly from the chat content"""
    try:
        drive_service = build('drive', 'v3', credentials=creds)
        
        # Read the uploaded file content into memory
        file_content = await pptx_file.read()

        # Define the metadata for the NEW file we want to create (a Google Slide)
        # We set the name and specify the target MIME type for Google Slides.
        file_metadata = {
            'name': pptx_file.filename.replace('.pptx', ''),
            'mimeType': 'application/vnd.google-apps.presentation'
        }

        # Prepare the media body for the upload.
        # We provide the raw content and specify the SOURCE MIME type (PowerPoint).
        # This tells Google Drive "take this PowerPoint file and convert it".
        media_body = MediaIoBaseUpload(
            io.BytesIO(file_content),
            mimetype='application/vnd.openxmlformats-officedocument.presentationml.presentation',
            resumable=True
        )

        # Execute the upload-and-convert request
        converted_slide = drive_service.files().create(
            body=file_metadata,
            media_body=media_body,
            fields='id, webViewLink' # Request the fields we need in the response
        ).execute()

        return {"presentationUrl": converted_slide.get('webViewLink')}

    except HttpError as error:
        detail = getattr(error, 'reason', str(error))
        raise HTTPException(status_code=400, detail=f"Google API Error: {detail}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
@router.post("/export-to-google-slides")
async def export_to_google_slides(presentation_data: PresentationData, creds: google.oauth2.credentials.Credentials = Depends(get_user_credentials)):
    """route called from the frontend in order to export chat contents to a slide show"""
    try:
        service = build('slides', 'v1', credentials=creds)

        # 1. Create a new blank presentation
        presentation = service.presentations().create(body={'title': presentation_data.title}).execute()
        presentation_id = presentation.get('presentationId')

        requests = []
        for i, slide_data in enumerate(presentation_data.slides):
            slide_id = f"slide_{i}"
            requests.append({'createSlide': {'objectId': slide_id, 'slideLayoutReference': {'predefinedLayout': 'BLANK'}}})

            # 2. Add Title
            title_id = f"title_{i}"
            requests.append({
                "createShape": {"objectId": title_id, "shapeType": "TEXT_BOX", "elementProperties": {"pageObjectId": slide_id, "size": {"height": {"magnitude": 50, "unit": "PT"}, "width": {"magnitude": 600, "unit": "PT"}}, "transform": {"scaleX": 1, "scaleY": 1, "translateX": 60, "translateY": 20, "unit": "PT"}}}
            })
            requests.append({"insertText": {"objectId": title_id, "text": slide_data.title, "insertionIndex": 0}})
            requests.append({"updateTextStyle": {"objectId": title_id, "style": {"fontSize": {"magnitude": 24, "unit": "PT"}, "bold": True}, "textRange": {"type": "ALL"}, "fields": "fontSize,bold"}})


            # 3. Add Image
            if slide_data.image:
                image_id = f"image_{i}"
                requests.append({
                    "createImage": {"objectId": image_id, "url": slide_data.image, "elementProperties": {"pageObjectId": slide_id, "size": {"height": {"magnitude": 250, "unit": "PT"}, "width": {"magnitude": 450, "unit": "PT"}}, "transform": {"scaleX": 1, "scaleY": 1, "translateX": 135, "translateY": 80, "unit": "PT"}}}
                })

            # 4. Add Summary Text
            summary_id = f"summary_{i}"
            requests.append({
                "createShape": {"objectId": summary_id, "shapeType": "TEXT_BOX", "elementProperties": {"pageObjectId": slide_id, "size": {"height": {"magnitude": 100, "unit": "PT"}, "width": {"magnitude": 600, "unit": "PT"}}, "transform": {"scaleX": 1, "scaleY": 1, "translateX": 60, "translateY": 350, "unit": "PT"}}}
            })
            requests.append({"insertText": {"objectId": summary_id, "text": slide_data.summary, "insertionIndex": 0}})
            requests.append({"updateTextStyle": {"objectId": summary_id, "style": {"fontSize": {"magnitude": 12, "unit": "PT"}}, "textRange": {"type": "ALL"}, "fields": "fontSize"}})

        # 5. Execute all requests in a single batch
        body = {'requests': requests}
        service.presentations().batchUpdate(presentationId=presentation_id, body=body).execute()

        return {"presentationUrl": f"https://docs.google.com/presentation/d/{presentation_id}/edit"}

    except HttpError as error:
        raise HTTPException(status_code=400, detail=f"Failed to create Google Slides presentation: {error.reason}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")