#Script to find cik number for the company.

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import re
import json
from google import genai
from google.genai import types


# === Gemini LLM Call Function ===

def llm_call(prompt):
    client = genai.Client(api_key="AIzaSyCXJ-rv6PZXnbyLQQ1eT9ek43Pb2UYzL6Q")

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

prompt_template ='''
**Role:** You are an AI assistant specialized in retrieving corporate financial information, specifically SEC CIK numbers and identifying corporate parent structures, prioritizing reliable data sources.

**Objective:** Given a list of specific company names, iterate through each name. For each company, determine its Central Index Key (CIK) number as registered with the U.S. Securities and Exchange Commission (SEC). If a provided company is not directly publicly traded or doesn't have its own distinct CIK (e.g., it's a subsidiary, brand, or division), identify its ultimate publicly traded parent company in the US and provide the CIK number for that parent. If neither the specified company nor its ultimate parent (if one exists) is publicly traded in the US, indicate this status. Compile the results for all processed companies into a single JSON list.

**Input:**
* `company_name_list`: A list of company names for which the CIK is required. Example: `["[Company Name 1]", "[Company Name 2]", ...]`

**Process Steps:**

Iterate through each `company_name` in the input `company_name_list`. For **each** `company_name`:

1.  **Initial Company Identification:** Accurately identify the specific company represented by the current `company_name` from the list.
2.  **Direct Public Check & CIK Retrieval:**
    * Determine if this identified company is itself a publicly traded entity in the US with its own SEC CIK number.
    * **Source Priority for CIK:** Primarily consult data consistent with the **SEC EDGAR database** (available at `www.sec.gov/edgar/searchedgar/cik.htm` or general EDGAR search). Use official CIK lookup tools or knowledge derived directly from SEC filings.
    * If **YES**, retrieve this CIK number. Note this result for the current company.
3.  **Parent Company Identification (If Necessary):**
    * If the initial company is **NOT** publicly traded *or* does not have its own distinct CIK: Identify its ultimate parent company. Focus on the parent entity that is most likely to be publicly traded in the US and file with the SEC.
    * **Source Priority for Parent Identification:** Consult information derived from sources such as:
        * **SEC filings:** Check Exhibit 21 ("Subsidiaries of the Registrant") in the parent company's 10-K annual reports if a potential parent is known. Search EDGAR for filings mentioning the subsidiary.
        * **Official Company Investor Relations websites:** Look for corporate structure information or lists of subsidiaries/brands.
        * **Reputable Financial Data Providers:** Utilize knowledge consistent with data from sources like Bloomberg, Refinitiv Eikon, S&P Capital IQ, FactSet.
        * **Reliable Financial News Reporting & Business Databases:** Cross-reference with established sources like Reuters, Wall Street Journal, Bloomberg News, Forbes, etc.
        * **Wikipedia/Wikidata:** Can be used as a starting point but should be verified against more authoritative sources listed above, especially for financial details.
4.  **Parent Company Public Check & CIK Retrieval:**
    * Determine if the identified parent company is publicly traded in the US and has an SEC CIK number.
    * **Source Priority for Parent's CIK:** Primarily consult data consistent with the **SEC EDGAR database** for the identified parent company's CIK confirmation.
    * If **YES**, retrieve the CIK number of this *parent* company. Note this result and the parent's name for the current company.
5.  **Handle Private Status:**
    * If the initial company is not public **AND** EITHER no parent company is found **OR** the identified ultimate parent company (based on reliable sources) is *also* not publicly traded in the US (and thus has no CIK), then the status for the current company is 'privately owned'. Note this result.

After processing all names in the `company_name_list`, compile the individual results into a single JSON list.

**Output Format:**

Provide the final result as a single JSON list containing a dictionary for *each* company processed from the input `company_name_list`. Each dictionary within the list must strictly follow the format below, ensuring the keys are exactly as shown ('company name', 'parent company', 'cik') and the values accurately reflect the findings for that specific company:

```json
[
  {{
    "company name": "<The first company name processed>",
    "parent company": "<Name of the identified parent company for the first company, IF applicable. Use 'N/A' or leave blank otherwise>",
    "cik": "<CIK number for the first company OR its parent OR 'privately owned'>"
  }},
  {{
    "company name": "<The second company name processed>",
    "parent company": "<Name of the identified parent company for the second company, IF applicable. Use 'N/A' or leave blank otherwise>",
    "cik": "<CIK number for the second company OR its parent OR 'privately owned'>"
  }},
  // ... one dictionary for each company in the input list
]

**Example:

Input: company_name_list = ["YouTube", "Microsoft", "Cargill"]
Expected Output:
JSON

[
  {{
    "company name": "YouTube",
    "parent company": "Alphabet Inc.",
    "cik": "1652044"
  }},
  {{
    "company name": "Microsoft",
    "parent company": "N/A",
    "cik": "789019"
  }},
  {{
    "company name": "Cargill",
    "parent company": "N/A",
    "cik": "privately owned"
  }}
]

**Key Considerations:

Source Prioritization: Strive to base findings, especially CIK numbers and parent relationships for public entities, on information verifiable through the SEC EDGAR database or highly reputable financial data sources/official company disclosures.
Accuracy is paramount for each company processed.
Distinguish between brands/divisions and actual legal entities registered with the SEC for each company.
The CIK should belong to the entity that files reports with the SEC.
If a parent exists but is also privately owned, the final output for cik for that company should be 'privately owned'.
Strictly adhere to the requested list-of-dictionaries JSON output format.
Now, execute this process for the following input:

{company_name_list}

'''

INPUT_CSV = "unique_parent_names.csv"
OUTPUT_CSV = "updated_cik.csv"

# Expected columns in the DataFrame
COMP_COL = "AI_derived_parent_name"  # The input company name
PARENT_COL = "GROUP_NAME"  # The parent company (to be determined)
CIK_COL = "cik"                # The retrieved CIK or "privately owned"

CHUNK_SIZE = 25
MAX_WORKERS = 12


# -------------------------------------------------------------------------------------------------
# 4) Main Function
# -------------------------------------------------------------------------------------------------
def main():
    """
    Main entry point for reading the CSV, splitting data into chunks,
    calling the LLM, and updating the CSV in parallel threads.
    """
    start = time.time()

    # 4.1) Load Data
    try:
        df = pd.read_csv(INPUT_CSV)
    except FileNotFoundError as e:
        print(f"❌ Could not find file '{INPUT_CSV}': {e}")
        return
    except Exception as e:
        print(f"❌ Error reading '{INPUT_CSV}': {e}")
        return

    # Ensure parent_company and cik columns exist
    if PARENT_COL not in df.columns:
        df[PARENT_COL] = None
    if CIK_COL not in df.columns:
        df[CIK_COL] = None

    # 4.2) Filter rows that still need processing
    unprocessed_df = df[df[CIK_COL].isna()]

    # print(unprocessed_df)

    # If there's nothing to process, no concurrency needed
    if unprocessed_df.empty:
        print("✅ No rows to process. The CSV may already be up-to-date.")
        return

    # 4.3) Split into chunks of size CHUNK_SIZE
    chunks = [unprocessed_df.iloc[i : i + CHUNK_SIZE] for i in range(0, len(unprocessed_df), CHUNK_SIZE)]

    # print(chunks[0], ' *******' , chunks[1])

    # 4.4) Function to process a chunk
    def process_chunk(chunk_df: pd.DataFrame):
        """
        Builds the prompt for all company names in chunk_df and calls the LLM.

        Args:
            chunk_df (pd.DataFrame): A subset of the original DataFrame.

        Returns:
            list of dict: The JSON response from LLM (each dict has "company name", "parent company", "cik")
        """
        comp_names = chunk_df[COMP_COL].tolist()
        # print(comp_names)
        # formatted_list = "\n".join(f'"{name}"' for name in comp_names)
        prompt = prompt_template.format(company_name_list=comp_names)
        llm_res = llm_call(prompt) 
        return llm_res

    # 4.5) Run concurrency
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]

        for i, future in enumerate(as_completed(futures), start=1):
            try:
                result = future.result()  # Should be a list of dict from the LLM
                if result and isinstance(result, list):
                    # Build an update map from "company name" -> { "parent_company": ..., "cik": ... }
                    update_map = {}
                    for entry in result:
                        if "company name" in entry:
                            comp = entry["company name"]
                            par = entry.get("parent company", "N/A")
                            cik = entry.get("cik", "privately owned")
                            update_map[comp] = {
                                "parent_company": par,
                                "cik": cik
                            }

                    # Update DataFrame rows where unique_comp_name matches
                    mask = df[COMP_COL].isin(update_map.keys())

                    # Update both parent_company and cik
                    df.loc[mask, PARENT_COL] = df.loc[mask, COMP_COL].apply(
                        lambda x: update_map[x]["parent_company"]
                    )
                    df.loc[mask, CIK_COL] = df.loc[mask, COMP_COL].apply(
                        lambda x: update_map[x]["cik"]
                    )

                    # Save partial progress after each chunk
                    try:
                        df.to_csv(OUTPUT_CSV, index=False)
                        print(f"✅ Saved after chunk {i}/{len(chunks)}", flush=True)
                    except Exception as e:
                        print(f"❌ Error saving CSV after chunk {i}: {e}", flush=True)
                else:
                    print(f"⚠️ Empty or invalid result for chunk {i}", flush=True)

            except Exception as e:
                print(f"❌ Error processing chunk {i}: {e}", flush=True)

    end = time.time()
    print(f"\n✅ All batches processed in {end - start:.2f} seconds.")
    print(f"✅ Final output written to {OUTPUT_CSV}")

# -------------------------------------------------------------------------------------------------
# 5) Entry Point
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
