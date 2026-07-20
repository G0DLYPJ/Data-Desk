# 📊 The Data Desk

> An intelligent, schema-aware data analysis workspace powered by Gemini 2.5.

**The Data Desk** translates plain-English analytical questions into executable Pandas, Matplotlib, and Seaborn code. Built with an emphasis on token efficiency and structured prompt engineering, it inspects your dataset's structural profile to generate clean, targeted code snippets.

---

## 🛠️ Key Features

* **Data Sanitization (`sanitize_dataframe`):** Automatically cleans raw DataFrame headers using regex to prevent execution-time Pandas syntax errors (trims spaces, strips special characters, normalizes delimiters).
* **Token-Efficient Schema Profiling (`get_data_profile`):** Extracts dataset metadata, statistical summaries, data types, and head previews without overwhelming the LLM's context window.
* **Strict Code-Generation System Prompt (`generate_code`):** Constrains Gemini to output direct, executable Python code without conversational overhead, forced to operate safely on in-memory DataFrame structures.
* **Context Awareness & Error Recovery:** Preserves multi-turn conversation context and accepts error tracebacks to iteratively self-correct code.

---

## 📁 Repository Structure

```text
├── engine.py           # Core analytics engine (Sanitization, Profiling, LLM Prompt Logic)
├── .env.example        # Environment variable template
├── .gitignore          # Excluded environment and cache files
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
🚀 Quickstart & Setup
1. Prerequisites
Ensure you have Python 3.10+ installed and a Google Gemini API key from Google AI Studio.

2. Clone & Environment Configuration
Bash
# Clone repository
git clone [https://github.com/G0DLYPJ/Data-Desk-.git](https://github.com/G0DLYPJ/Data-Desk-.git)
cd Data-Desk-

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
3. API Key Setup
Copy .env.example to create your local .env file:

Bash
cp .env.example .env
Open .env and insert your Gemini API Key:

Code snippet
GEMINI_API_KEY=your_actual_api_key_here
💡 Engine Workflow
Plaintext
  [ Raw CSV / DataFrame ]
             │
             ▼
   sanitize_dataframe()  ──> Clean column names & drop empty columns
             │
             ▼
   get_data_profile()    ──> Generate token-efficient metadata profile
             │
             ▼
    generate_code()      ──> Query Gemini 2.5 Flash with strict system constraints
             │
             ▼
  [ Executable Python String ]
