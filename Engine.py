import io
import re
import pandas as pd
from dotenv import load_dotenv
from google import genai

# Load environment variables (API keys, etc.)
load_dotenv()
client = genai.Client()

def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans column names to prevent Pandas syntax errors during dynamic code execution."""
    df = df.copy()
    
    # Drop completely empty columns upfront
    df = df.dropna(how='all', axis=1)
    
    # Clean up column names: replace special chars and extra spaces with single underscores
    clean_cols = []
    for col in df.columns:
        c = str(col).strip()
        c = re.sub(r'[^\w\s]', '_', c)
        c = re.sub(r'\s+', '_', c)
        c = re.sub(r'_+', '_', c)
        clean_cols.append(c)
        
    df.columns = clean_cols
    return df

def get_data_profile(df: pd.DataFrame) -> str:
    """Extracts a lightweight metadata profile to give the LLM context without passing the entire dataset."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    # Cap sample preview to 10 columns max so prompt tokens stay reasonable
    sample_df = df.iloc[:, :10] if df.shape[1] > 10 else df
    
    profile = f"""
### DATASET PROFILE ###
- Total Rows: {df.shape[0]}
- Total Columns: {df.shape[1]}

- Column Information:
{info_str}

- Statistical Summary:
{df.describe(include='all').to_string()}

- First 5 Rows Preview:
{sample_df.head(5).to_string()}
"""
    return profile

def generate_code(user_query: str, data_profile: str, chat_history: list = None, error_msg: str = None) -> str:
    """Generates pure executable Python code using Gemini based on dataset schema and user intent."""
    
    # System prompt forces strict code-only output format
    system_instruction = """
    You are an expert Python Data Analyst and GenAI Agent.
    You are provided with metadata and sample rows of a pandas DataFrame named `df`.
    Your primary goal is to answer the user's question by generating EXECUTABLE Python code using `pandas`, `numpy`, `matplotlib`, or `seaborn`.

    STRICT GUIDELINES:
    1. Assume `df` is ALREADY loaded in memory. Do NOT try to reload or redefine `df`.
    2. Write ONLY valid, executable Python code without markdown code fences.
    3. Do NOT import pandas, os, sys, or load data from disk.
    4. For text/numbers/summaries: ALWAYS print them out clearly using print().
    5. For visualizations: Use seaborn or matplotlib. Always add titles/labels. End with `plt.tight_layout()`. Do NOT call `plt.show()`.
    6. Ensure column names match the dataset profile EXACTLY.
    """

    # Retain recent conversation context (last 6 messages)
    history_str = ""
    if chat_history:
        history_str = "\n### RECENT CHAT HISTORY ###\n"
        recent_messages = chat_history[-6:]
        for msg in recent_messages:
            role = "User" if msg.get("role") == "user" else "Assistant Insight/Result"
            content = msg.get("content", "")
            history_str += f"- {role}: {content[:200]}\n"

    prompt = f"DATASET PROFILE:\n{data_profile}\n{history_str}\nUSER QUESTION: {user_query}"
    
    # Append execution trace if retrying after a failed run
    if error_msg:
        prompt += f"\n\nCRITICAL FIX NEEDED:\nYour previous attempt failed with this error:\n{error_msg}\nReview the profile carefully and write corrected code."

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"system_instruction": system_instruction}
    )
    
    # Clean up markdown code blocks if the model accidentally includes them
    code = response.text.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
        
    return code.strip()