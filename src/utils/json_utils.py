import json
import re

def robust_parse_json(raw_text: str):
    """
    Try to robustly parse Gemini output into valid JSON.
    Handles cases with markdown fences, trailing commas, and natural language wrappers.
    Returns (parsed_json or None, debug_message)
    """
    if not raw_text or not isinstance(raw_text, str):
        return None, "Empty or non-string output"

    text = raw_text.strip()

    # --- 1️⃣ remove markdown code block wrappers ---
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # --- 2️⃣ extract JSON-like substring ---
    match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
    if match:
        text = match.group(1)

    # --- 3️⃣ try direct parse ---
    try:
        return json.loads(text), "Parsed directly"
    except json.JSONDecodeError as e:
        last_err = str(e)

    # --- 4️⃣ attempt to repair common syntax errors ---
    repaired = text

    # 4.1 remove trailing commas before ] or }
    repaired = re.sub(r",\s*(\]|\})", r"\1", repaired)

    # 4.2 balance braces/brackets if missing one at end
    if repaired.count("{") > repaired.count("}"):
        repaired += "}"
    if repaired.count("[") > repaired.count("]"):
        repaired += "]"

    # 4.3 try parse again
    try:
        return json.loads(repaired), "Parsed after repair"
    except json.JSONDecodeError as e:
        last_err = str(e)

    # --- 5️⃣ ultimate fallback: try to extract multiple JSONs concatenated ---
    json_like_parts = re.findall(r"(\{[^{}]+\})", repaired)
    if json_like_parts:
        try:
            arr = [json.loads(p) for p in json_like_parts]
            return arr, "Parsed partial JSON array"
        except Exception:
            pass

    return None, f"Failed to parse JSON: {last_err}"
