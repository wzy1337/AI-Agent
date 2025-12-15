# 🔭 Horizon Scanning AI Agent

A GenAI-powered horizon scanning system for monitoring policy issues to automatically extract relevant events from online sources.

## What It Does

This system performs **automated horizon scanning** to identify emerging issues, weak signals, and global trends that could impact CPF and/or its members thereby assisting policymakers in decision making
Does the following:
- **Searches** across many curated query categories (global crises, tech disruption, weak signals, Singapore-specific)
- **Analyzes** findings using GPT-4o to extract structured insights
- **Deduplicates** against previous reports to surface only new information
- **Generates** PDF reports and JSON exports
- **Stores** results in Google Sheets for cloud deployment


## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        config.py                                │
│              Define queries, LLM settings, thresholds           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
══════════════════════════════╪═══════════════════════════════════
  STAGE 1: KNOWLEDGE GATHERING
══════════════════════════════╪═══════════════════════════════════
                              ▼
              ┌───────────────────────────────┐
              │  main.py                      │
              │  - Stage 1 LLM prompt         │
              │  - Agent setup & tools        │──→ tools.py
              └───────────────┬───────────────┘    (Tavily, DuckDuckGo)
                              │
                              ▼
              ┌───────────────────────────────┐
              │  stage1_knowledge.py          │
              │  - Execute 13 queries         │
              │  - URL extraction & filtering │──→ utils/date_utils.py
              │  - Collect raw results        │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Raw Knowledge Base           │
              │  (URLs, snippets, metadata)   │
              └───────────────┬───────────────┘
                              │
══════════════════════════════╪═══════════════════════════════════
  STAGE 2: LLM ANALYSIS & STRUCTURING
══════════════════════════════╪═══════════════════════════════════
                              ▼
              ┌───────────────────────────────┐
              │  main.py                      │
              │  - Stage 2 LLM prompt         │
              │  - GPT-4o analyzes knowledge  │──→ models.py
              │  - Extract structured events  │    (Pydantic schemas to ensure desired structured output)
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Deduplication & Filtering    │──→ utils/file_utils.py
              │  Compare with past reports    │
              │  Remove duplicates            │
              └───────────────┬───────────────┘
                              │
══════════════════════════════╪═══════════════════════════════════
  OUTPUT & PRESENTATION
══════════════════════════════╪═══════════════════════════════════
                              ▼
              ┌───────────────────────────────┐
              │  Generate JSON + PDF          │──→ utils/pdf_export.py
              │  Save to Google Sheets        │──→ cloud_storage.py
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  app.py - Streamlit Dashboard │
              │  View reports, trigger scans  │
              └───────────────────────────────┘
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/wzy1337/AI-Agent.git
cd AI-Agent
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `sample.env` file in the project root and add your keys:

```env
OPENAI_API_KEY="sk-..."
TAVILY_API_KEY="tvly-..."
```

### 3. Run Locally

**Option A: Streamlit Dashboard**
```bash
streamlit run app.py
```

**Option B: Command Line**
```bash
python main.py
```

## ☁️ Streamlit Cloud Deployment

### Step 1: Push to GitHub
Push your code to GitHub (make sure API keys are NOT in your code!)

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your repo: `wzy1337/AI-Agent`
4. Set main file path: `app.py`
5. Click **Deploy**

### Step 3: Configure Secrets
Go to your app → **Settings** → **Secrets** and add:

```toml
# Required API Keys
OPENAI_API_KEY = "sk-..."
TAVILY_API_KEY = "tvly-..."
# Sample Json key file
[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "123456789"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"

[google_sheets]
spreadsheet_id = "your-spreadsheet-id-from-url"

# Google Sheets Integration
# 1. Go to Google Cloud Console → Create new project
# 2. Enable Google Sheets API & Google Drive API
# 3. Create Service Account- API & Services Credentials -> Create Credntials-> Service Account
# 4. Create API Key: Click on service account-> Under 'key' section -> Add key-> Create new key->Json
# 5. Download JSON key file
# 6. Share your Google Sheet with the service account email
# 7. Add relevant details onto streamlit secrets below


```

> 💡 **Note:** Get the `spreadsheet_id` from your Google Sheet URL:  
> URL:`https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit`

## 📁 Project Structure

```
AI-Agent/
├── app.py                 # Streamlit dashboard
├── main.py                # Core research agent
├── stage1_knowledge.py    # Knowledge query execution
├── config.py              # Queries & settings
├── models.py              # Pydantic data models
├── tools.py               # Tavily, Duckduckgo tools
├── cloud_storage.py       # Google Sheets integration
├── utils/
│   ├── date_utils.py      # Date extraction & filtering
│   ├── file_utils.py      # File operations
│   └── pdf_export.py      # PDF report generation
├── requirements.txt       # Installed packages
└── sample.env             # Template for API keys
```

## ⚙️ Configuration

Edit `config.py` to customize:

| Setting | Description |
|---------|-------------|
| `LLM_MODEL` | OpenAI model (default: `gpt-4o`) |
| `FUZZY_MATCH_THRESHOLD` | Similarity threshold for deduplication (0.7 = 70%) |
| `KNOWLEDGE_QUERIES_TIER` | List of horizon scanning queries |

## 🔍 Query Categories

The system searches across International, Regional and Local covering think pieces, new articles:

1. **Global Crises & Warnings** - IMF/OECD alerts, worldwide pension issues
2. **Reform Experiments** - Nordic, Netherlands, UK, Australia innovations
3. **Asian Challenges** - Japan, Korea, Taiwan, HK, Malaysia aging
4. **Tech Disruption** - AI, gig economy, fintech impacts
5. **Weak Signals** - Reddit, fringe movements, anomalies
6. **Adjacent Domains** - Housing, healthcare, ESG, generational issues
7. **Singapore-Specific** - CPF debates, local research
8. **Expert Opinions** - Economist predictions, sustainability warnings

## 📊 Output Formats

- **JSON** - Structured data for programmatic use
- **PDF** - Formatted reports for stakeholders
- **Google Sheets** - Cloud storage for historical tracking

## 🔧 API Requirements

| API | Purpose | Get Key |
|-----|---------|---------|
| OpenAI | GPT-4o for analysis | [platform.openai.com](https://platform.openai.com) |
| Tavily | Web search & extraction | [tavily.com](https://tavily.com) |
| Google Sheets | Cloud storage | [Google Cloud Console](https://console.cloud.google.com) |

## 📸 Dashboard Preview

![Dashboard](images/Dashboard.png)

## 📖 How to Use

1. Click **"Start Research Job"** to run a new scan
2. Wait ~15-20 minutes for research to complete
3. Once done, the new report will be displayed
4. Repeated events from previous scans are highlighted

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Submit a pull request
