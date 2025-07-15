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

import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt
# if "google.colab" in sys.modules:
#     from google.colab import auth

#     auth.authenticate_user()


PROJECT_ID = "dl-test-439308"  # 

MODEL_ID = "gemini-2.5-flash"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")

# Concurrency settings for API calls
MAX_CONCURRENT_REQUESTS = 5 # That could be increased 
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

SYSTEM_INSTRUCTION = f"""
    You are an expert assistant designed to evaluate the usefulness of comments for generating highly specific and 
    actionable climate and disaster mitigation recommendations for urban settings.

    Your task is to determine if a given comment provides valuable, contextual information that can lead to more precise 
    and location-specific mitigation actions.

    You should know that all comments are geolocalized and are part of a larger dataset that includes general and local 
    context about flood risk, fire risk and heat risk.

    A comment is considered **useful (1)** if it offers:
    - Specific local context about community initiatives, past events, unique vulnerabilities, or specific opportunities 
        (e.g., community interest in green infrastructure, specific areas prone to flooding, existing local groups).
    - Information that helps tailor mitigation actions to specific demographics or points of interest 
        (e.g., details about access issues for community members at-risk, specific building vulnerabilities, 
        underutilized spaces).
    - Insights into socio-economic factors that could influence the feasibility or type of recommendations.
    - Details about past successes, failures, or specific impacts of climate events in the area.

    A comment is considered **not useful (0)** if it:
    - Is generic or lacks specificity (e.g., "it gets hot here," "some people are old").
    - Refers to isolated, non-generalizable, or trivial past events without broader implications 
        (e.g., "it rained last Tuesday").
    - Does not provide any actionable insight for mitigation planning.
    - Is redundant with information already present in the general or local context.

    Your output must be in JSON format.

    """ 

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(10))
async def filter_comments(row_id, comment, print_output=False):

    async with semaphore:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION, http_options=HttpOptions(api_version="v1"))

        PROMPT_TEMPLATE = f"""
            Please evaluate the usefulness of the following comment for generating actionable climate and disaster 
            mitigation recommendations in an urban setting. Respond in JSON format with a 'useful' key 
            (1 for useful, 0 for not useful) and an 'explanation' key.

            Comment: '{comment}'
        """

        try:
            # Generate the structured response
            response = await client.aio.models.generate_content(
                model=MODEL_ID,
                contents=[PROMPT_TEMPLATE],
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.4,
                    top_p=0.95,
                    top_k=20,
                    candidate_count=1,
                    seed=5,  # ALWAYS SAME ANSWERS!
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                ),
            )

            if response.text:
                if print_output:
                    print("System Instruction:", SYSTEM_INSTRUCTION)
                    print("Prompt Template:", PROMPT_TEMPLATE)
                    print("Response:", response.text)
                
                return row_id, response.text
            else:
                # Handle cases where response.text might be empty or problematic
                print(f"Warning: Empty response for row_id {row_id}, comment: '{comment}'. Full response: {response}")
                return row_id, "No useful information found."

        
        except Exception as e:
            print(f"Error generating content for row_id {row_id}, prompt: '{prompt}': {e}")
            return row_id, f"Error: {e}"
      

        
