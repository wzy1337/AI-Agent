An GenAI-powered horizon scanning system for monitoring policy issues to automatically extract relevat events from online source. 

## What It Does

This system performs **automated horizon scanning** to identify emerging issues, weak signals, and global trends that could impact retirement systems. 
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
              └───────────────┬───────────────┘    (Tavily, Wikipedia)
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

Copy the sample environment file and add your keys:

```bash
cp sample.env .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
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

1. Push your code to GitHub (without API keys!)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your repo
4. Add secrets in **Settings → Secrets**:
   ```toml
   OPENAI_API_KEY = "sk-..."
   TAVILY_API_KEY = "tvly-..."
   gcp_service_account="" download json from GCP
   Spreadsheet_id="____" 


   ```

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
| `MIN_YEAR_FILTER` | Only include sources from this year onwards |
| `FUZZY_MATCH_THRESHOLD` | Similarity threshold for deduplication (0.7 = 70%) |
| `KNOWLEDGE_QUERIES_TIER` | List of horizon scanning queries |

## 🔍 Query Categories

The system searches across these tiers:

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
| Google Sheets (optional) | Cloud storage | [Google Cloud Console](https://console.cloud.google.com) |

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch
3. Submit a pull request