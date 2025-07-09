from IPython.display import HTML, Markdown, display
from google import genai
from google.genai.types import (        
    HttpOptions,
    GenerateContentConfig,
)
import os
import pandas as pd
import geopandas as gpd
import sys
from pydantic import BaseModel
from typing import List

# if "google.colab" in sys.modules:
#     from google.colab import auth

#     auth.authenticate_user()


PROJECT_ID = "dl-test-439308"  # 

MODEL_ID = "gemini-2.5-flash"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION, http_options=HttpOptions(api_version="v1"))

if not client.vertexai:
    print("Using Gemini Developer API.")
elif client._api_client.project:
    print(
        f"Using Vertex AI with project: {client._api_client.project} in location: {client._api_client.location}"
    )
elif client._api_client.api_key:
    print(
        f"Using Vertex AI in express mode with API key: {client._api_client.api_key[:5]}...{client._api_client.api_key[-5:]}"
    )


def filter_comments(comment, print_output=False):

    SYSTEM_INSTRUCTION = f"""
    You are an expert assistant designed to evaluate the usefulness of comments for generating highly specific and actionable climate and disaster mitigation recommendations for urban settings.

    Your task is to determine if a given comment provides valuable, contextual information that can lead to more precise and location-specific mitigation actions.

    You should know that all comments are geolocalized and are part of a larger dataset that includes general and local context about climate risks, flood risk, fire risk and heat risk.

    A comment is considered **useful (1)** if it offers:
    - Specific local context about community initiatives, past events, unique vulnerabilities, or specific opportunities (e.g., community interest in green infrastructure, specific areas prone to flooding, existing local groups).
    - Information that helps tailor mitigation actions to specific demographics or points of interest (e.g., details about access issues for community members at-risk, specific building vulnerabilities, underutilized spaces).
    - Insights into socio-economic factors that could influence the feasibility or type of recommendations.
    - Details about past successes, failures, or specific impacts of climate events in the area.

    A comment is considered **not useful (0)** if it:
    - Is generic or lacks specificity (e.g., "it gets hot here," "some people are old").
    - Refers to isolated, non-generalizable, or trivial past events without broader implications (e.g., "it rained last Tuesday").
    - Does not provide any actionable insight for mitigation planning.
    - Is redundant with information already present in the general or local context.

    Your output must be in JSON format.

    """ 


    PROMPT_TEMPLATE = f"""
    Please evaluate the usefulness of the following comment for generating actionable climate and disaster mitigation recommendations in an 
    urban setting. Respond in JSON format with a 'useful' key (1 for useful, 0 for not useful) and an 'explanation' key.

    Comment: '{comment}'
    """


    #Generate the structured response
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[PROMPT_TEMPLATE],
        config=GenerateContentConfig(
            response_mime_type="application/json",
            #response_schema=RiskActions if not explain else RiskActions_and_explanation,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.4,
            top_p=0.95,
            top_k=20,
            candidate_count=1,
            seed=5, #ALWAYS SAME ANSWERS!
            #max_output_tokens=50, # could be useful to limit the output length (as its not needed)
            presence_penalty=0.0,
            frequency_penalty=0.0,
        ),
    )

    if print_output:
        print("System Instruction:", SYSTEM_INSTRUCTION)
        print("Prompt Template:", PROMPT_TEMPLATE)
        print("Response:", response.text)


    return response.text
