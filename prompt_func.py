from IPython.display import HTML, Markdown, display
from google import genai
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

# if "google.colab" in sys.modules:
#     from google.colab import auth

#     auth.authenticate_user()


PROJECT_ID = "dl-test-439308"  # 
# if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
#     PROJECT_ID = str(os.environ.get("GOOGLE_CLOUD_PROJECT"))
MODEL_ID = "gemini-2.5-flash-preview-05-20" #"gemini-2.0-flash"  # @param {type: "string"}
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



def generate_risk_actions(municipality_context, heat_risk, flood_risk, fire_risk, lst_day, lst_night,
                          sealed_surface_pct, canopy_cover_pct, elevation, river_proximity,
                          flood_plain, tree_count, flammability, tree_connectivity,
                          fire_history_info, population_density, vulnerable_groups, pois, climate_driven_impassable_roads,
                          emergency_assemble_areas, comments, pdf_uri=None, explain=False, print_output=False
                          ):


    OUTUT_RISK = """    
    Output only the action plan.
    Format: 3 sections: 🔥 Fire, 🌡 Heat, 🌊 Flood. Max 3 actions each. Only if risk is relevant. Each action = 1 bullet, 1–2 lines.
    """

    OUTPUR_RISK_EXPLAIN = """    
    Output the solutions and explain (chain of thoughts) why you propose this solution refering to the inputs you considered.
    Format: 2 times 3 sections: 🔥 Fire, 🌡 Heat, 🌊 Flood. Max 3 actions each. Only if risk is relevant. Each solution = 1 bullet, 1–2 lines. explanation for that solution in its seperate section also following the bullets
    """

    SYSTEM_INSTRUCTION = f"""
    You are an environmental expert creating actionable mitigation recommendations for climate and disaster resilience in urban settings.

    Below is a description of the current risks and environmental conditions for a municipality and a specific local zone. You are tasked with generating mitigation actions tailored to that zone's risks, demographic context, and relevant points of interest.

    Use clear language and address actions that can be taken both by individuals and by local authorities. Use bullet points where helpful. Be specific, not generic. 
    When referring to community members, use 'community members at-risk' instead of terms like 'elderly' or 'vulnerable' that may not be appropriate in all contexts.

    🎯 TASK:
    Based on this information, list **concise practical and location-specific mitigation actions** that can reduce climate and disaster risks in the local area. Structure them by risk type (e.g., Heat, Flood, Fire). Include targeted suggestions related to nearby POIs or vulnerable populations.

    {OUTUT_RISK if not explain else OUTPUR_RISK_EXPLAIN}
    """ #     When referencing assembly areas, always refer as 'community nominated assembly area' instead of using their names


    PROMPT_TEMPLATE = f"""

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



    #Read pdf document if url provided
    # if pdf_uri:
    #     PDF_FILE = Part.from_uri(
    #         file_uri=pdf_uri,
    #         mime_type="application/pdf",
    #     )

    PDF_FILES = None
    if pdf_uri:
        PDF_FILES = []

        if isinstance(pdf_uri, list):
            PDF_FILES = [
                Part.from_uri(file_uri=uri, mime_type="application/pdf")
                for uri in pdf_uri
            ]
        else:
            PDF_FILES = [Part.from_uri(file_uri=pdf_uri, mime_type="application/pdf")]



    #Generate the structured response
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[PDF_FILES, PROMPT_TEMPLATE] if PDF_FILES else [PROMPT_TEMPLATE],
        config=GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=RiskActions if not explain else RiskActions_and_explanation,
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.4,
            top_p=0.95,
            top_k=20,
            candidate_count=1,
            seed=5, #ALWAYS SAME ANSWERS!
            # max_output_tokens=100,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        ),
    )

    if print_output:
        print("System Instruction:", SYSTEM_INSTRUCTION)
        print("Prompt Template:", PROMPT_TEMPLATE)
        # print("Response Schema:", RiskActions.schema_json(indent=2))
        print("Response:", response.text)


    return response.text



map_data = {
    'heat_risk':['high risk', 'highly susceptible', 'medium risk', 'low risk'],
    'flood_risk' :['high risk', 'highly susceptible', 'medium risk', 'low risk'],
    'fire_risk' : ['high risk', 'highly susceptible', 'medium risk', 'low risk'],  
}  