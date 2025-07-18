from IPython.display import HTML, Markdown, display
from google import genai
from google.genai import types
from google.genai.types import (        
    HttpOptions,
    FunctionDeclaration,
    GenerateContentConfig,
    GoogleSearch,
    HarmBlockThreshold,
    HarmCategory,
    MediaResolution,
    Part,
    Retrieval,
    SafetySetting,
    Tool,
    ToolCodeExecution,
    VertexAISearch,
)
import os
import pandas as pd
import geopandas as gpd
import sys
from pydantic import BaseModel
from typing import List
OUTPUT_RISK_JSON = '{"fire": ["string"], "heat": ["string"], "flood": ["string"]}'
OUTPUT_RISK_EXPLAIN_JSON = '{"fire": ["string"], "explanation_fire": ["string"], "heat": ["string"], "explanation_heat": ["string"], "flood": ["string"], "explanation_flood": ["string"]}'

import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt
# if "google.colab" in sys.modules:
#     from google.colab import auth

#     auth.authenticate_user()


PROJECT_ID = "dl-test-439308"

MODEL_ID = "gemini-2.5-flash"
LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", "global")

# Concurrency settings for API calls
MAX_CONCURRENT_REQUESTS = 5 # That could be increased 
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

OUTPUT_RISK = """    
    Output only the action plan.
    Format: 3 sections: 🔥 Fire, 🌡 Heat, 🌊 Flood. Max 3 actions each. Only if risk is relevant. 
    Each action = 1 bullet, 1–2 lines.
    """

OUTPUT_RISK_EXPLAIN = """    
    Output the solutions and explain (chain of thoughts) why you propose this solution refering to the inputs you considered.
    Format: 2 times 3 sections: 🔥 Fire, 🌡 Heat, 🌊 Flood. Max 3 actions each. Only if risk is relevant. 
    Each solution = 1 bullet, 1–2 lines. explanation for that solution in its seperate section also following the bullets
    """

OUTPUT_RISK_JSON = '{"fire": ["string"], "heat": ["string"], "flood": ["string"]}'
OUTPUT_RISK_EXPLAIN_JSON = '{"fire": ["string"], "explanation_fire": ["string"], "heat": ["string"], "explanation_heat": ["string"], "flood": ["string"], "explanation_flood": ["string"]}'

# Step 1: Define the output schema
class RiskActions(BaseModel):
    fire: List[str]
    heat: List[str]
    flood: List[str]

class RiskActions_and_explanation(BaseModel):
    fire:List[str]
    explanation_fire:List[str]
    heat:List[str]
    explanation_heat:List[str]
    flood:List[str]
    explanation_flood:List[str]

def cache_files(pdf_uri, explain=False, time='20000s'):

    SYSTEM_INSTRUCTION = prepare_system_prompt(explain) # When referencing assembly areas, always refer as 'community nominated assembly area' instead of using their names

    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION, http_options=HttpOptions(api_version="v1"))


    PDF_FILES = prepare_files(pdf_uri)

    cache = client.caches.create(
        model=MODEL_ID,
        config=types.CreateCachedContentConfig(
            display_name='dargo_files_SystemPrompt', # used to identify the cache
            system_instruction=SYSTEM_INSTRUCTION,
            contents=[PDF_FILES],
            ttl=time,
        )
    )

    return cache

async def generate_risk_actions(row_id, municipality_context, heat_risk, flood_risk, fire_risk, lst_day, lst_night,
                          sealed_surface_pct, canopy_cover_pct, elevation, river_proximity,
                          flood_plain, tree_count, flammability, tree_connectivity,
                          fire_history_info, population_density, vulnerable_groups, pois, climate_driven_impassable_roads,
                          emergency_assemble_areas, comments, pdf_uri=None, synthesis_pdf=None, cache=None, explain=False, print_output=False
                          ):

    async with semaphore:
        client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION, http_options=HttpOptions(api_version="v1"))

        PROMPT_TEMPLATE = prepare_prompt_risk_action(
            municipality_context=municipality_context,
            heat_risk=heat_risk,
            flood_risk=flood_risk,
            fire_risk=fire_risk,
            lst_day=lst_day,
            lst_night=lst_night,
            sealed_surface_pct=sealed_surface_pct,
            canopy_cover_pct=canopy_cover_pct,
            elevation=elevation,
            river_proximity=river_proximity,
            flood_plain=flood_plain,
            tree_count=tree_count,
            flammability=flammability,
            tree_connectivity=tree_connectivity,
            fire_history_info=fire_history_info,
            population_density=population_density,
            vulnerable_groups=vulnerable_groups,
            pois=pois,
            climate_driven_impassable_roads=climate_driven_impassable_roads,
            emergency_assemble_areas=emergency_assemble_areas,
            comments=comments,
            pdf_uri=pdf_uri, #synthesis_pdf, #pdf_uri, # synthesis_pdf, # cache=cached_files, 
            cache=cache, 
            print_output=print_output
        )


        try:
            #Generate the structured response
            response = await client.aio.models.generate_content(
                model=MODEL_ID,
                #contents=[PDF_FILES, PROMPT_TEMPLATE] if PDF_FILES else [PROMPT_TEMPLATE],
                contents=[PROMPT_TEMPLATE],
                config=GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RiskActions if not explain else RiskActions_and_explanation,
                    #system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.4,
                    top_p=0.95,
                    top_k=20,
                    candidate_count=1,
                    seed=5, #ALWAYS SAME ANSWERS!
                    # max_output_tokens=100,
                    presence_penalty=0.0,
                    frequency_penalty=0.0,
                    cached_content=cache.name
                ),
            ) ## thinking can be limited 

            if response.text:
                if print_output:
                    #print("System Instruction:", SYSTEM_INSTRUCTION)
                    print("Prompt Template:", PROMPT_TEMPLATE)
                    # print("Response Schema:", RiskActions.schema_json(indent=2))
                    print("Response:", response.text)
                    print("Response JSON:", response)

                return row_id, response.text
            else:
                # Handle cases where response.text might be empty or problematic
                print(f"Warning: Empty response for prompt: '{PROMPT_TEMPLATE}'. Full response: {response}")
                return row_id, "No useful information found."

        except Exception as e:
            print(f"Error row {row_id} generating content for prompt: '{PROMPT_TEMPLATE}': {e}")
            return row_id, f"Error: {e}"

def prepare_prompt_risk_action(municipality_context, heat_risk, flood_risk, fire_risk, lst_day, lst_night,
                          sealed_surface_pct, canopy_cover_pct, elevation, river_proximity,
                          flood_plain, tree_count, flammability, tree_connectivity,
                          fire_history_info, population_density, vulnerable_groups, pois, climate_driven_impassable_roads,
                          emergency_assemble_areas, comments, pdf_uri=None, synthesis_pdf=None, cache=None, print_output=False
                          ):
    prompt = f"""

            INPUT
            ---
            🟩 GENERAL CONTEXT (Municipality-Level):
            # {municipality_context}

            🟨 LOCAL CONTEXT (Buffer zone of ~500m radius):
            - Heat Risk (monthly): {heat_risk}
            - Flood Risk: {flood_risk}
            - Fire Risk (monthly): {fire_risk}
            - LST Day/Night: {lst_day}/{lst_night}
            - Land Surface: Sealed Surface = {sealed_surface_pct}%
            - Canopy Cover = {canopy_cover_pct}%
            - Elevation = {elevation}m
            - River proximity = {river_proximity} m
            - Floodplain: {flood_plain}
            - Tree cover: Number of trees = {tree_count}
            - Flammability Index = {flammability}
            - Connectivity = {tree_connectivity}
            - Fire history: {fire_history_info}


            📍 POINTS OF INTEREST:
            - Nearby: {pois}
            - climate_driven_impassable_roads: {climate_driven_impassable_roads}
            - emergency_assemble_areas: {emergency_assemble_areas}
            ---

            COMMENTS:
            {comments}
        """
    return prompt

def prepare_system_prompt(explain=False):
    
    system_instruction = f"""
    You are an environmental expert creating actionable mitigation recommendations for climate and disaster resilience in urban settings.

    Below is a description of the current risks and environmental conditions for a municipality and a specific local zone. You are tasked with generating mitigation actions tailored to that zone's risks, demographic context, and relevant points of interest.

    Use clear language and address actions that can be taken both by individuals and by local authorities. Use bullet points where helpful. Be specific, not generic. 
    When referring to community members, use 'community members at-risk' instead of terms like 'elderly' or 'vulnerable' that may not be appropriate in all contexts.

    🎯 TASK:
    Based on this information, list **concise practical and location-specific mitigation actions** that can reduce climate and disaster risks in the local area. Structure them by risk type (e.g., Heat, Flood, Fire). Include targeted suggestions related to nearby POIs or vulnerable populations.

    {OUTPUT_RISK if not explain else OUTPUT_RISK_EXPLAIN}
    """  
    return system_instruction

def prepare_system_prompt_batching(explain=False):
    
    system_instruction = """
    You are an environmental expert creating actionable mitigation recommendations for climate and disaster resilience in 
    urban settings.

    Below is a description of the current risks and environmental conditions for a municipality and a specific local zone. 
    You are tasked with generating mitigation actions tailored to that zone's risks, demographic context, 
    and relevant points of interest.

    Use clear language and address actions that can be taken both by individuals and by local authorities. 
    Use bullet points where helpful. Be specific, not generic. 
    When referring to community members, use 'community members at-risk' instead of terms like 'elderly' or 'vulnerable' 
    that may not be appropriate in all contexts.

    ---
    **STRICT FORMATTING INSTRUCTIONS: YOUR ONLY OUTPUT IS THE RAW JSON OBJECT. NO OTHER TEXT, NO MARKDOWN, NO EXPLANATIONS OUTSIDE THE JSON.**
    1.  **OUTPUT MUST BE A SINGLE, RAW, VALID JSON OBJECT.**
    2.  **DO NOT INCLUDE ANY TEXT OR CHARACTERS OUTSIDE THE JSON OBJECT.**
    3.  **DO NOT WRAP THE JSON OBJECT IN ANY QUOTES (e.g., no `"` at the very start/end).**
    4.  **ABSOLUTELY DO NOT USE MARKDOWN CODE BLOCKS (e.g., NO ```json, NO ```).**
    5.  **THE VERY FIRST CHARACTER OF YOUR OUTPUT MUST BE `{`.**
    6.  **THE VERY LAST CHARACTER OF YOUR OUTPUT MUST BE `}`.**
    7.  **ALL STRING VALUES WITHIN THE JSON MUST BE VALID JSON STRINGS (properly escaped quotes, no extraneous newlines unless explicitly allowed by JSON string rules).**
    8.  **USE STANDARD ASCII BULLET POINTS `- ` OR `* ` IF LISTS ARE DESIRED WITHIN STRINGS, AND ENSURE THEY ARE PART OF A VALID JSON STRING.**
    9.  **ENSURE ALL KEYS AND VALUES IN THE JSON STRICTLY ADHERE TO THE PROVIDED SCHEMA.**
    ---

    """ + (OUTPUT_RISK_JSON if not explain else OUTPUT_RISK_EXPLAIN_JSON) + """

    🎯 TASK:
    Based on this information, list **concise practical and location-specific mitigation actions** that can reduce climate 
    and disaster risks in the local area. Structure them by risk type (e.g., Heat, Flood, Fire). 
    Include targeted suggestions related to nearby POIs or vulnerable populations.

    Ensure each action is a concise string. Do not use complex formatting or excessive newlines within the action strings themselves beyond what is strictly necessary.

    """ + (OUTPUT_RISK if not explain else OUTPUT_RISK_EXPLAIN) + """
    """
    return system_instruction

def prepare_files(pdf_uri):
    if pdf_uri:
        PDF_FILES = []

        if isinstance(pdf_uri, list):
            PDF_FILES = [
                Part.from_uri(file_uri=uri, mime_type="application/pdf") for uri in pdf_uri
            ]
        else:
            PDF_FILES = [Part.from_uri(file_uri=pdf_uri, mime_type="application/pdf")]

        return PDF_FILES
    
    else:
        return None

def prepare_files_for_jsonl(pdf_uri):
    """
    Prepares file data in the format required for JSONL batch prediction.
    """
    if pdf_uri:
        file_data_list = []
        if isinstance(pdf_uri, list):
            for uri in pdf_uri:
                file_data_list.append({
                    "file_uri": uri,
                    "mime_type": "application/pdf"
                })
        else:
            file_data_list.append({
                "file_uri": pdf_uri,
                "mime_type": "application/pdf"
            })
        return file_data_list
    else:
        return None

def prepare_prompt_systemprompt_files_batch(row_id, municipality_context, heat_risk, flood_risk, fire_risk, lst_day, lst_night,
                          sealed_surface_pct, canopy_cover_pct, elevation, river_proximity,
                          flood_plain, tree_count, flammability, tree_connectivity,
                          fire_history_info, population_density, vulnerable_groups, pois, climate_driven_impassable_roads,
                          emergency_assemble_areas, comments, pdf_uri=None, synthesis_pdf=None, cache=None, explain=False, print_output=False):
    pdf_files = prepare_files_for_jsonl(pdf_uri)
    system_instruction = prepare_system_prompt_batching(explain)
    prompt = prepare_prompt_risk_action(
            municipality_context=municipality_context,
            heat_risk=heat_risk,
            flood_risk=flood_risk,
            fire_risk=fire_risk,
            lst_day=lst_day,
            lst_night=lst_night,
            sealed_surface_pct=sealed_surface_pct,
            canopy_cover_pct=canopy_cover_pct,
            elevation=elevation,
            river_proximity=river_proximity,
            flood_plain=flood_plain,
            tree_count=tree_count,
            flammability=flammability,
            tree_connectivity=tree_connectivity,
            fire_history_info=fire_history_info,
            population_density=population_density,
            vulnerable_groups=vulnerable_groups,
            pois=pois,
            climate_driven_impassable_roads=climate_driven_impassable_roads,
            emergency_assemble_areas=emergency_assemble_areas,
            comments=comments,
            pdf_uri=pdf_uri, #synthesis_pdf, #pdf_uri, # synthesis_pdf, # cache=cached_files, 
            cache=cache,  
            print_output=print_output
        )
    
    return row_id, pdf_files, system_instruction, prompt

map_data = {
    'heat_risk':['high risk', 'highly susceptible', 'medium risk', 'low risk'],
    'flood_risk' :['high risk', 'highly susceptible', 'medium risk', 'low risk'],
    'fire_risk' : ['high risk', 'highly susceptible', 'medium risk', 'low risk'],  
}  