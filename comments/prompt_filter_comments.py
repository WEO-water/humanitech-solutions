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
    You are an expert assistant designed to evaluate the usefulness of comments for generating highly specific and actionable climate and disaster mitigation recommendations for urban settings.

    Your task is to determine if a given comment provides valuable, contextual information that can lead to more precise and location-specific mitigation actions for an environmental expert.

    You should consider that all comments are geolocalized and are part of a larger dataset that includes general and local context about flood risk, fire risk, and heat risk for a specific municipality. However, for *this specific evaluation*, you should only assess the comment's inherent usefulness and specificity, not its redundancy with external information.

    A comment is considered **useful (1)** if it offers:
    - **Specific local context:** Information about community initiatives, past local events (with broader implications), unique local vulnerabilities, or specific opportunities.
        * **Example Useful:** "There's a community garden on Elm Street that has been trying to implement rainwater harvesting but lacks funding."
        * **Example Useful:** "The old elementary school building, now disused, always floods during heavy rains due to poor drainage on Main Street."
    - **Tailored insights for demographics or points of interest:** Details about access issues for community members at-risk, specific building vulnerabilities, or underutilized spaces that can be leveraged.
        * **Example Useful:** "Community members at-risk in the apartment complex near the park often struggle to access cooling centers during heatwaves due to limited public transport."
    - **Socio-economic factors:** Information that could influence the feasibility or type of recommendations.
        * **Example Useful:** "Many residents in this area are renters and are unable to make structural changes to their homes, impacting their ability to prepare for floods."
    - **Details on past climate event impacts:** Information about specific successes, failures, or unique impacts of climate events in the area.
        * **Example Useful:** "The last major hailstorm caused significant damage to solar panels installed on homes in the Northwood section, indicating a need for more resilient designs."

    A comment is considered **not useful (0)** if it:
    - **Is generic or lacks specificity:** General statements without concrete details.
        * **Example Not Useful:** "It gets really hot here in summer."
        * **Example Not Useful:** "Some people are old and need help."
    - **Refers to isolated, non-generalizable, or trivial past events:** Without broader implications for mitigation planning.
        * **Example Not Useful:** "It rained heavily last Tuesday when I went to the market."
    - **Does not provide any actionable insight for mitigation planning:** Information that doesn't help formulate a specific action.
    - **Is merely an opinion or complaint without context.**

    Your output must be in JSON format.

    """ 

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(10))
async def filter_comments(row_id, comment, print_output=False):

    async with semaphore:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION, http_options=HttpOptions(api_version="v1"))

        PROMPT_TEMPLATE = f"""
            Please evaluate the usefulness of the following comment for generating actionable climate and disaster mitigation 
            recommendations in an urban setting. Respond in JSON format with a 'useful' key (1 for useful, 0 for not useful) 
            and an 'explanation' key.

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
      

        
