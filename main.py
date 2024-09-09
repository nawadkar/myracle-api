from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from typing import List, Optional, Dict, Any
import google.generativeai as genai
import base64
import json
import re
import pandas as pd
from io import BytesIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://myracle-case-study-pfkzkjpld-nawadkars-projects.vercel.app",
        "http://localhost:3000"  # Allow localhost for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Gemini API
genai.configure(api_key = os.getenv("GEMINI_API_KEY"))  # Replace with your actual API key

class TestStep(BaseModel):
    step_count: int = Field(..., description="The step number of the test case.")
    step_description: str = Field(..., description="The description of the test step.")

class Feature(BaseModel):
    description: str = Field(description="What the test case is about.")
    pre_conditions: str = Field(description="What needs to be set up or ensured before testing.")
    testing_steps: List[TestStep] = Field(description="Clear, step-by-step instructions on how to perform the test.")
    expected_result: str = Field(description="The expected result of the feature")

class FeatureList(BaseModel):
    features: List[Feature]

    def to_dict(self):
        return {"features": [feature.model_dump() for feature in self.features]}

    def to_csv(self):
        rows = []
        for i, feature in enumerate(self.features, 1):
            base_row = {
                "Feature Number": i,
                "Description": feature.description,
                "Pre-conditions": feature.pre_conditions,
                "Expected Result": feature.expected_result
            }
            for step in feature.testing_steps:
                row = base_row.copy()
                row.update({
                    "Step Number": step.step_count,
                    "Step Description": step.step_description
                })
                rows.append(row)
        return pd.DataFrame(rows)

def encode_image(image_file: bytes):
    return base64.b64encode(image_file).decode('utf-8')

def clean_response(response: str):
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        return match.group(0)
    return response

def preprocess_response(response: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    if "featureList" in response:
        return {"features": response["featureList"]}
    elif "features" in response:
        return response
    else:
        raise ValueError("Unexpected response structure")

def generate_test_instructions(images: List[bytes], context: Optional[str] = None):
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": encode_image(img)
        } for img in images
    ]

    parser = PydanticOutputParser(pydantic_object=FeatureList)
    format_instructions = parser.get_format_instructions()
    
    prompt = """You are an AI assistant that generates detailed and professional testing instructions for digital products.
    You will be given a list of images that show different features of a digital product.
    You will need to generate testing instructions for each feature.
    """

    if context:
        prompt += f"\nContext: {context}"

    prompt += f"\nThe response should be in JSON format. \n {format_instructions}"
    
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content([prompt] + image_parts)
    results = clean_response(response.text)
    return results

@app.post("/generate_instructions/")
async def generate_instructions(
    files: List[UploadFile] = File(...),
    context: Optional[str] = Form(None)
):
    if len(files) < 5 or len(files) > 10:
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload between 5 and 10 screenshots."}
        )
    
    images = [await file.read() for file in files]
    response = generate_test_instructions(images, context)
    
    try:
        response_dict = json.loads(response)
        # preprocessed_response = preprocess_response(response_dict)
        feature_list = FeatureList.model_validate(response_dict)
        return {
            # "features": feature_list.to_dict(),
            "features": feature_list.to_dict(),
            "csv": feature_list.to_csv().to_dict(orient="records")
            # "csv": feature_list.to_csv()
        }
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Unable to parse the generated instructions. Details: {str(e)}"}
        )
    except ValueError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected response structure: {str(e)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"An error occurred: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
