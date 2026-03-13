import base64
import json
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from litellm import completion

app = FastAPI()

SUPPORTED_MIME_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpg",
    "application/pdf": "pdf",
}


PROMPT = """Analyze this document and extract all structured information from it.
Return the output as a valid JSON object that captures all the key data, fields, and information present in the document.
The JSON structure should be dynamic and appropriate for the document type (invoice, receipt, form, ID card, etc.).
Only return the JSON object, no additional text or explanation."""


@app.post("/extract")
async def extract_document(file: UploadFile = File(...)):
    if file.content_type not in SUPPORTED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported types: {list(SUPPORTED_MIME_TYPES.keys())}"
        )

    content = await file.read()
    base64_content = base64.b64encode(content).decode("utf-8")
    media_type = file.content_type

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{base64_content}"
                    }
                },
                {
                    "type": "text",
                    "text": PROMPT
                }
            ]
        }
    ]

    try:
        response = completion(model="ollama/deepseek-ocr", messages=messages)
        result_text = response.choices[0].message.content
        print(result_text)
        try:
            if result_text.startswith("```json"):
                result_text = result_text[7:]
            if result_text.startswith("```"):
                result_text = result_text[3:]
            if result_text.endswith("```"):
                result_text = result_text[:-3]
            result_text = result_text.strip()
            
            extracted_data = json.loads(result_text)
        except json.JSONDecodeError:
            extracted_data = {"raw_text": result_text}

        return {"extracted_data": extracted_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
