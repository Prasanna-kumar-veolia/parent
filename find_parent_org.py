#Script to find the actual parent company.

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import json
from google import genai
from google.genai import types


# === Gemini LLM Call Function ===

def llm_call(prompt):
    client = genai.Client(api_key="AIzaSyCFNniHyMbebPMsHArfhbP3E52PPC4vl-g")

    tools = [
        types.Tool(google_search=types.GoogleSearch())
    ]

    generate_content_config = types.GenerateContentConfig(
        tools=tools,
        response_mime_type="text/plain",
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=generate_content_config
    )

    pattern = re.compile(r'\[.*?\]', re.DOTALL)
    match = pattern.search(response.text)
    if match:
        json_str = match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("❌ JSON decoding error:", e, flush=True)
            return []
    else:
        print("⚠️ No JSON block found in response", flush=True)
        return []


# === Prompt Template ===

prompt_template = '''
You are an intelligent assistant tasked with identifying the **ultimate parent company** for each facility in the given list. A facility may belong to a private company, a brand, a government agency, or a municipal corporation.

Your goal is to determine, for each facility, the **top-level entity** that owns or governs it (i.e., the ultimate parent company). Follow the steps below for each facility:

1. **Normalize the facility name** by removing generic suffixes such as “Plant”, “Facility”, etc. 
2. **Perform a targeted web search/ Google search** using **GoogleSearch()** tool for queries like:
   - "parent company for <FACILITY NAME>"
   - "Who owns <FACILITY NAME>"
   - "<FACILITY NAME> owned by"
3. Pay special attention to the FACILITY NAME, Utilize the web search results to the correct parent company.
3. **Check official or brand websites** for corporate ownership details.
4. **Search business directories** such as OpenCorporates, Crunchbase, or Bloomberg.
5. **If the facility is government/municipal**, use queries like "FACILITY NAME site:.gov".
6. **Cross-reference public datasets** like the EPA FRS or Data.gov.
7. **Disambiguate** facilities with similar names using context.
8. **"Not Found" Criteria:**
    * Return "parent company name": "Not Found" only after thoroughly checking at least 5 credible sources, including official company websites, SEC filings, and reputable business databases.
    * Do not stop searching until all logical search avenues are exhausted.

**Output Format:**
Return the results strictly as a JSON array with the following structure:
[
{{
"facility name": "<input facility name 1>",
"parent company name": "<identified parent company 1>"
}},
{{
"facility name": "<input facility name 2>",
"parent company name": "<identified parent company 2>"
}},
...
]

Return only the JSON output — no additional text, commentary, or formatting outside the JSON block.

**Input List:**
{facility_list}
'''

# === Config ===

INPUT_CSV = "epa_final_data_epa_org_names.csv"
OUTPUT_CSV = "updated_parent_1.csv"
FACILITY_COL = "FAC_NAME"
RESULT_COL = "AI_derived_parent_name"
CHUNK_SIZE = 50
MAX_WORKERS = 20

# === Load Data ===

df = pd.read_csv(INPUT_CSV)

# Create the column if not exists
if RESULT_COL not in df.columns:
    df[RESULT_COL] = None

# Filter unprocessed rows
unprocessed_df = df[df[RESULT_COL].isna()]

# Split into chunks of 50
chunks = [unprocessed_df.iloc[i:i + CHUNK_SIZE] for i in range(0, len(unprocessed_df), CHUNK_SIZE)]


# === Batch Processing Function ===

def process_chunk(chunk_df):
    facilities = chunk_df[FACILITY_COL].tolist()
    formatted_list = "\n".join(f'"{name}"' for name in facilities)
    prompt = prompt_template.format(facility_list=formatted_list)
    result = llm_call(prompt)
    return result


# === Run Concurrently ===

start = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

    for i, future in enumerate(as_completed(futures)):
        try:
            result = future.result()
            if result:
                update_map = {
                    entry["facility name"]: entry["parent company name"]
                    for entry in result if "facility name" in entry and "parent company name" in entry
                }

                mask = df[FACILITY_COL].isin(update_map.keys()) & df[RESULT_COL].isna()
                df.loc[mask, RESULT_COL] = df.loc[mask, FACILITY_COL].map(update_map)

                # Save to CSV after each chunk
                df.to_csv(OUTPUT_CSV, index=False)
                print(f"✅ Saved after chunk {i+1}/{len(chunks)}", flush=True)
            else:
                print(f"⚠️ Empty result from chunk {i+1}", flush=True)

        except Exception as e:
            print(f"❌ Error processing chunk {i+1}: {e}", flush=True)

end = time.time()

print(f"\n✅ All batches processed in {end - start:.2f} seconds.", flush=True)
print(f"✅ Output written to {OUTPUT_CSV}", flush=True)