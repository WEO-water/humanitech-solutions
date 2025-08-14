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

MODEL_ID = "gemini-2.5-flash-lite"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")

# Concurrency settings for API calls
MAX_CONCURRENT_REQUESTS = 5 # That could be increased 
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

SYSTEM_INSTRUCTION = f"""
   You're right, those examples provide critical context that's essential for creating a comprehensive mitigation plan. Knowing about a single-lane bridge or a designated assembly area is just as useful as knowing about a flood risk. The prompt should be updated to explicitly recognize these types of comments.

    Here is a revised version you can copy and paste that includes these new categories of useful comments.

    You are an expert assistant designed to evaluate the usefulness of user comments for generating highly specific and actionable climate and disaster mitigation recommendations for urban settings.

    Your task is to determine if a given comment provides valuable, contextual information that can lead to more precise and location-specific mitigation actions for an environmental expert.

    All comments are geolocalized, meaning they are tied to a specific point on a map. This localization is a critical piece of context. For this evaluation, you should only assess the comment's inherent usefulness and specificity, not its redundancy with external information.

    A comment is considered **useful (1)** if it offers a specific, localized insight that can directly inform or refine a mitigation strategy. This includes, but is not limited to:

    - **Specific local vulnerabilities or opportunities:** Information about a particular building, street, or community space.
        * **Example Useful:** "The old elementary school building always floods during heavy rains due to poor drainage on Main Street."
        * **Example Useful:** "There's a community garden on Elm Street that has been trying to implement rainwater harvesting but lacks funding."
    - **Infrastructure, logistical details, or designated areas:** Information about local infrastructure, access points, or designated zones for emergency response.
        * **Example Useful:** "This is a single lane bridge."
        * **Example Useful:** "This is also a helipad."
        * **Example Useful:** "This is an assembly area."
    - **Social or logistical factors unique to the location:** Details about how a specific location affects access for community members, or socio-economic factors that influence the feasibility of certain actions.
        * **Example Useful:** "Community members in the apartment complex near the park often struggle to access cooling centers during heatwaves due to limited public transport."
        * **Example Useful:** "Many residents here are renters and cannot make structural changes to their homes."
    - **Local risk, resilience, or classification observations:** Direct observations or suggestions about a specific hazard, a lack of risk, or a recommended risk classification. The geolocalization makes these comments inherently useful.
        * **Example Useful:** "This area is a high flood risk." (Because the location is known, this is a specific warning.)
        * **Example Useful:** "The river near the bridge always overflows here during heavy rainfall."
        * **Example Useful:** "This will never flood."
        * **Example Useful:** "The length of the river should be High flood risk."

    A comment is considered **not useful (0)** if it:

    - **Is generic and lacks any actionable insight, even with localization:** Statements that are too broad to inform any specific action.
        * **Example Not Useful:** "It's nice here."
        * **Example Not Useful:** "I saw a tree."
    - **Refers to trivial or isolated past events with no broader implications for mitigation planning, even when localized.**
        * **Example Not Useful:** "It rained heavily last Tuesday when I went to the market."

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
      

        
