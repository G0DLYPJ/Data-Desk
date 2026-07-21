import io
import os
import re
import ast
import contextlib
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from google import genai

# ReportLab imports for PDF export
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

load_dotenv()
client = genai.Client()


def check_code_safety(code_str: str) -> tuple[bool, str]:
    """
    AST Security Parser: Inspects code structure BEFORE execution.
    Blocks forbidden imports, system calls, and file I/O operations.
    """
    FORBIDDEN_MODULES = {'os', 'sys', 'subprocess', 'shutil', 'builtins', 'socket', 'requests', 'urllib'}
    FORBIDDEN_FUNCTIONS = {'exec', 'eval', 'open', 'compile', '__import__'}

    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, f"Syntax Error in generated code: {e}"

    for node in ast.walk(tree):
        # Check forbidden imports (e.g., import os, from os import path)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split('.')[0] in FORBIDDEN_MODULES:
                    return False, f"Security Violation: Forbidden import '{alias.name}' detected."
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split('.')[0] in FORBIDDEN_MODULES:
                return False, f"Security Violation: Forbidden import from '{node.module}' detected."

        # Check forbidden function calls (e.g., open(), eval())
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_FUNCTIONS:
                    return False, f"Security Violation: Forbidden function call '{node.func.id}()' detected."

    return True, ""


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans column names to prevent Pandas syntax errors."""
    df = df.copy()
    df = df.dropna(how='all', axis=1)
    
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
    """Extracts a concise metadata profile from the DataFrame."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    
    sample_df = df.iloc[:, :10] if df.shape[1] > 10 else df
    
    profile = f"""
### DATASET PROFILE ###
- Total Rows: {df.shape[0]}
- Total Columns: {df.shape[1]}

- Column Information:
{info_str}

- Statistical Summary:
{df.describe(include='all').to_string()}

- First 5 Rows:
{sample_df.head(5).to_string()}
"""
    return profile


def generate_code(user_query: str, data_profile: str, chat_history: list = None, error_msg: str = None) -> str:
    """Generates Python code based on dataset metadata and recent chat context."""
    system_instruction = """
    You are an expert Python Data Analyst and GenAI Agent.
    You are provided with metadata and sample rows of a pandas DataFrame named `df`.
    Your primary goal is to answer the user's question by generating EXECUTABLE Python code using `pandas`, `numpy`, `matplotlib`, or `seaborn`.

    STRICT GUIDELINES:
    1. Assume `df` is ALREADY loaded in memory. Do NOT try to reload or redefine `df`.
    2. Write ONLY valid, executable Python code. Do NOT output markdown code fences (no ```python or ```).
    3. Do NOT import pandas, os, sys, or load data from disk.
    4. For text, numbers, summaries, or insights: ALWAYS print them out clearly using print().
    5. For visualizations (charts/plots):
       - Use seaborn or matplotlib.
       - Always add titles and labels for clarity.
       - End with `plt.tight_layout()`.
       - Do NOT call `plt.show()`.
    6. Ensure column names match the dataset profile EXACTLY (case-sensitive).
    7. Consider previous user questions and context if the query is a follow-up.
    """

    history_str = ""
    if chat_history:
        history_str = "\n### RECENT CHAT HISTORY ###\n"
        recent_messages = chat_history[-6:]
        for msg in recent_messages:
            role = "User" if msg.get("role") == "user" else "Assistant Insight/Result"
            content = msg.get("content", "")
            history_str += f"- {role}: {content[:200]}\n"

    prompt = f"DATASET PROFILE:\n{data_profile}\n{history_str}\nUSER QUESTION: {user_query}"
    
    if error_msg:
        prompt += f"\n\nCRITICAL FIX NEEDED:\nYour previous attempt failed with this error:\n{error_msg}\nReview the profile carefully and write corrected code."

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"system_instruction": system_instruction}
    )
    
    code = response.text.strip()
    if code.startswith("```python"):
        code = code[9:]
    elif code.startswith("```"):
        code = code[3:]
    if code.endswith("```"):
        code = code[:-3]
        
    return code.strip()


def execute_code(code_str: str, df: pd.DataFrame):
    """
    Executes generated Python code safely after passing AST security check.
    """
    # 1. AST Security Gatekeeper Check
    is_safe, safety_msg = check_code_safety(code_str)
    if not is_safe:
        return None, None, safety_msg

    local_env = {
        'df': df,
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns
    }
    
    output_buffer = io.StringIO()
    plt.close('all')
    
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code_str, local_env)
        
        printed_output = output_buffer.getvalue().strip()
        fig = plt.gcf() if plt.get_fignums() else None
        
        return printed_output, fig, None

    except Exception as e:
        return None, None, str(e)


def generate_insight(user_query: str, text_result: str, has_fig: bool) -> str:
    """Generates a 1-2 sentence executive takeaway based on execution output."""
    prompt = f"""
    The user asked: "{user_query}"
    The execution result output was:
    "{text_result if text_result else 'A visualization plot was generated.'}"

    Provide a crisp, 1-2 sentence executive takeaway or key observation based on this output.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text.strip()


def run_analyst_pipeline(user_query: str, raw_df: pd.DataFrame, chat_history: list = None, max_retries: int = 2):
    """Main pipeline: Sanitizes -> Profiles -> AST Checks -> Executes -> Returns Insight."""
    df = sanitize_dataframe(raw_df)
    profile = get_data_profile(df)
    error_msg = None
    attempted_code = ""

    for attempt in range(max_retries + 1):
        attempted_code = generate_code(user_query, profile, chat_history=chat_history, error_msg=error_msg)
        text_out, fig_out, error = execute_code(attempted_code, df)
        
        if not error:
            insight = generate_insight(user_query, text_out, fig_out is not None)
            
            return {
                "success": True,
                "code": attempted_code,
                "text_result": text_out,
                "fig_result": fig_out,
                "insight": insight,
                "sanitized_df": df,
                "attempts": attempt + 1
            }
        
        error_msg = error

    return {
        "success": False,
        "code": attempted_code,
        "error": error_msg,
        "sanitized_df": df,
        "attempts": max_retries + 1
    }


def generate_pdf_report(messages: list, filename: str = "Analysis_Report.pdf") -> str:
    """
    Compiles chat session Q&A, AI insights, and generated Seaborn figures into a clean PDF.
    Windows-safe file handling for temporary images.
    """
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=12,
        textColor=colors.HexColor('#1E293B')
    )
    
    question_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Heading2'],
        fontSize=12,
        spaceBefore=10,
        spaceAfter=6,
        textColor=colors.HexColor('#2563EB')
    )
    
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=8,
        leading=14
    )

    story = []
    story.append(Paragraph("📊 Conversational Data Analysis Report", title_style))
    story.append(Paragraph("Generated automatically by GenAI Data Analyst Agent", body_style))
    story.append(Spacer(1, 15))

    temp_files = []

    # Iterate through chat pairs
    for i in range(0, len(messages), 2):
        if i >= len(messages):
            break
            
        user_msg = messages[i]
        asst_msg = messages[i+1] if (i+1) < len(messages) else None

        if user_msg["role"] == "user":
            story.append(Paragraph(f"Q: {user_msg['content']}", question_style))

        if asst_msg and asst_msg["role"] == "assistant":
            story.append(Paragraph(f"<b>Insight:</b> {asst_msg['content']}", body_style))

            # Save figure to temporary image file to embed in PDF
            if asst_msg.get("figure") is not None:
                # Create a temp file path and close handle immediately so Windows releases the lock
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(tmp_fd)
                
                asst_msg["figure"].savefig(tmp_path, bbox_inches='tight', dpi=150)
                temp_files.append(tmp_path)
                story.append(RLImage(tmp_path, width=400, height=220))
                story.append(Spacer(1, 10))

        story.append(Spacer(1, 10))

    # Build PDF document
    doc.build(story)

    # Clean up temporary image files safely
    for path in temp_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass  # Ignore persistent Windows file locks so PDF generation succeeds

    return filename