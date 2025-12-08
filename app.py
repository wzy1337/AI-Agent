import streamlit as st
import json
import glob
import os
import subprocess
from datetime import datetime
from pathlib import Path
import time
# psutil is needed for reliable process checking, so we import it conditionally/here.
# Note: For conciseness, the complex check_research_status logic is mostly retained
# as it's crucial for managing the background process state safely.
try:
    import psutil
except ImportError:
    psutil = None

# Cloud storage for reading historical data
try:
    from cloud_storage import sheets_storage, IS_STREAMLIT_CLOUD
    CLOUD_ENABLED = True
except ImportError:
    CLOUD_ENABLED = False
    IS_STREAMLIT_CLOUD = False

# --- Configuration & CSS ---
CSS = """
<style>
    .main-header { font-size: 2.5rem; color: #1a5490; font-weight: bold; margin-bottom: 1rem; }
    .event-card { background-color: #f0f6ff; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #1a5490; margin-bottom: 1.5rem; }
    .event-title { font-size: 1.3rem; color: #1a5490; font-weight: bold; margin-bottom: 0.5rem; }
    .metadata-badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 15px; font-size: 0.85rem; margin-right: 0.5rem; margin-bottom: 0.5rem; color: white; }
    .stButton>button { background-color: #1a5490; color: white; font-weight: bold; border-radius: 5px; padding: 0.5rem 2rem; }
    .stButton>button:hover { background-color: #2c5f8d; }
    .running-indicator { background-color: #000000; padding: 1rem; border-radius: 5px; border-left: 5px solid #ffaa00; margin-bottom: 1rem; }

    /* Consolidated Badge Colors */
    .relevance-high { background-color: #ff4444; }
    .relevance-medium { background-color: #ffaa00; }
    .relevance-low { background-color: #44ff44; }
    .signal-established { background-color: #1a5490; }
    .signal-emerging { background-color: #ff8800; }
    .signal-weak { background-color: #888888; }
</style>
"""
st.set_page_config(page_title="CPF Research Intelligence Dashboard", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)


# --- Helper Functions (Consolidated) ---

RELEVANCE_MAP = {'high': 'relevance-high', 'medium': 'relevance-medium', 'med': 'relevance-medium'}
SIGNAL_MAP = {'established': 'signal-established', 'emerging': 'signal-emerging', 'weak': 'signal-weak'}

def load_research_data(filepath):
    """Load research data from JSON file or cloud storage."""
    # Check if this is a cloud reference
    if filepath.startswith("cloud:"):
        if CLOUD_ENABLED:
            try:
                report_id = filepath.replace("cloud:", "")
                events = sheets_storage.get_events_by_report(report_id)
                # Convert to expected format
                return {"events": [
                    {
                        "event": e.get("event_name", ""),
                        "description": e.get("description", ""),
                        "relevance": e.get("relevance", ""),
                        "signal_strength": e.get("signal_strength", ""),
                        "impact": e.get("impact", ""),
                        "scenario": e.get("scenario", ""),
                        "policy_intervention": e.get("policy_intervention", ""),
                        "location": e.get("location", ""),
                        "actors": e.get("actors", "").split(", ") if e.get("actors") else [],
                        "confidence": e.get("confidence", ""),
                        "source": e.get("sources", "").split(", ") if e.get("sources") else [],
                        "date": e.get("dates", "").split(", ") if e.get("dates") else []
                    } for e in events
                ]}
            except Exception as e:
                st.error(f"Error loading from cloud: {e}")
                return None
        return None
    
    # Local file
    try:
        return json.loads(Path(filepath).read_text(encoding='utf-8'))
    except Exception as e:
        st.error(f"Error loading {filepath}: {e}")
        return None

def get_badge_html(label, value, badge_type):
    """Generates HTML for a styled badge."""
    value_lower = value.split(' ')[0].lower() if isinstance(value, str) else str(value).lower()
    
    if badge_type == 'relevance':
        css_class = RELEVANCE_MAP.get(value_lower, 'relevance-low')
        display_label = f"Relevance: {value.split(' ')[0]}"
    elif badge_type == 'signal':
        css_class = SIGNAL_MAP.get(value_lower, 'signal-weak')
        display_label = value.split(' ')[0]
    else:
        css_class = 'signal-weak' # Default fallback
        display_label = f"{label}: {value}"
    
    return f'<span class="metadata-badge {css_class}">{display_label}</span>'

def get_all_research_files():
    """Get all research output JSON files sorted by date (newest first)."""
    local_files = sorted(glob.glob("research_output_*.json"), reverse=True)
    
    # On Streamlit Cloud, check Google Sheets for reports
    if CLOUD_ENABLED:
        try:
            cloud_reports = sheets_storage.get_all_reports(max_n=20)
            # Return report_ids as pseudo-filenames for compatibility
            cloud_files = [f"cloud:{r.get('report_id')}" for r in cloud_reports if r.get('report_id')]
            
            # If on cloud with no local files, use cloud files
            # If local files exist, combine them (local first, then cloud)
            if not local_files:
                return cloud_files
            else:
                # Combine: local files + cloud files not already in local
                local_ids = set(f.replace("research_output_", "").replace(".json", "") for f in local_files)
                unique_cloud = [f for f in cloud_files if f.replace("cloud:", "") not in local_ids]
                return local_files + unique_cloud
        except Exception as e:
            st.warning(f"Could not load from Google Sheets: {e}")
    
    return local_files

def parse_timestamp(filename):
    """Extract and format timestamp from filename."""
    try:
        ts_str = filename.replace('research_output_', '').replace('.json', '')
        return datetime.strptime(ts_str, '%Y%m%d_%H%M%S')
    except:
        return None

# The run_research_backend and check_research_status functions are critical 
# for the background process and are left largely as-is due to their complexity.
# A small consolidation is done in check_research_status:

def check_research_status():
    """Check if research is running, completed, or failed."""
    status_file = Path('research_status.json')
    if not status_file.exists(): return {'status': 'idle'}
    
    try:
        status_data = json.loads(status_file.read_text())
        
        # Stale "starting" status cleanup (retained for safety)
        if status_data.get('status') == 'starting':
            try:
                start_time = datetime.fromisoformat(status_data['start_time'])
                if (datetime.now() - start_time).seconds > 30:
                    status_file.unlink()
                    return {'status': 'idle'}
            except:
                status_file.unlink()
                return {'status': 'idle'}

        # Process check (retained for safety)
        if status_data.get('status') == 'running' and status_data.get('pid') and psutil:
            if not psutil.pid_exists(status_data['pid']):
                # Process finished, check log for success/error
                log_path = Path(status_data.get('log_file', ''))
                status_data['status'] = 'completed'
                if log_path.exists():
                    log_tail = log_path.read_text(encoding='utf-8', errors='ignore')[-1000:]
                    if any(err in log_tail for err in ['Traceback', 'Error:']):
                        status_data.update({'status': 'error', 'error': 'Process failed - check log file'})
                
                status_data['end_time'] = datetime.now().isoformat()
                status_file.write_text(json.dumps(status_data))
                return status_data
        
        return status_data
    except Exception:
        return {'status': 'idle'}

def run_research_backend():
    """Run the main.py research script in background using subprocess"""
    try:
        import sys
        
        # Create a status file to track progress
        status_file = Path('research_status.json')
        
        # Determine Python executable - use venv if available
        python_exe = sys.executable
        
        # Check if we're in a venv and use the venv python
        venv_python = Path('venv/Scripts/python.exe')
        if venv_python.exists():
            python_exe = str(venv_python.absolute())
        
        # For Windows, we need to use DETACHED_PROCESS to truly background it
        # Also redirect output to log files
        log_file = Path(f'research_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        
        # Create initial status before starting process
        status_file.write_text(json.dumps({
            'status': 'starting',
            'start_time': datetime.now().isoformat(),
            'log_file': str(log_file)
        }))
        
        # Open log file
        log_handle = open(log_file, 'w', encoding='utf-8')
        
        # Run python main.py in a truly detached subprocess
        if os.name == 'nt':  # Windows
            DETACHED_PROCESS = 0x00000008
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            process = subprocess.Popen(
                [python_exe, 'main.py'],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                cwd=os.getcwd(),
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                close_fds=True  # Ensure file handles are closed in child
            )
        else:  # Unix/Linux/Mac
            process = subprocess.Popen(
                [python_exe, 'main.py'],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=os.getcwd(),
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                close_fds=True
            )
        
        # Don't wait for the process - return immediately
        # The file handle will be closed by the subprocess
        
        # Update status file with PID and log file
        status_file.write_text(json.dumps({
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'pid': process.pid,
            'log_file': str(log_file)
        }))
        
        return True
        
    except Exception as e:
        status_file = Path('research_status.json')
        status_file.write_text(json.dumps({
            'status': 'error',
            'error': str(e),
            'end_time': datetime.now().isoformat()
        }))
        return False


# --- UI Components Refactor ---

def render_event_details(event):
    """Renders the full details of a single event inside an expander."""
    
    # Define fields to render inside the expander
    detail_fields = [
        ("Description", event.get('description', 'No description available')),
        ("Impact Analysis", event.get('impact', 'No impact analysis available')),
        ("Scenarios", event.get('scenario', 'No scenarios available')),
        ("Policy Intervention Recommendations", event.get('policy_intervention', 'No recommendations available'))
    ]

    with st.expander("📖 View Details", expanded=False):
        for title, content in detail_fields:
            st.markdown(f"**{title}:**")
            st.markdown(content)

        if event.get('informal_insights'):
            st.markdown("**Informal Insights (from forums, social media):**")
            st.info(event.get('informal_insights'))
        
        # Metadata Columns
        col1, col2 = st.columns(2)
        
        # Helper to display lists/strings
        def display_list_or_string(title, data):
            st.markdown(f"**{title}:**")
            if isinstance(data, list):
                if data:
                    for item in data: st.write(f"• {item}")
                else: st.write("N/A")
            elif data:
                 st.write(data)
            else:
                st.write("N/A")
        
        with col1:
            display_list_or_string("Location", event.get('location', 'N/A'))
            display_list_or_string("Actors", event.get('actors', []))
        with col2:
            display_list_or_string("Confidence", event.get('confidence', 'N/A'))
            display_list_or_string("Dates", event.get('date', []))
            
        # Sources
        st.markdown("**Sources:**")
        source_data = event.get('source', [])
        sources = source_data if isinstance(source_data, list) else [s.strip() for s in str(source_data).split(',')] if isinstance(source_data, str) else []
        
        for source in filter(None, sources):
            source = source.strip()
            if source.startswith('http'):
                st.markdown(f"• [{source[:80]}...]({source})")
            else:
                st.write(f"• {source}")

def render_event_card(idx, event):
    """Renders a single event card with badges and details expander."""
    
    # Event Card HTML
    st.markdown(f"""
    <div class="event-card">
        <div class="event-title">{idx}. {event.get('event', 'Untitled Event')}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Metadata badges
    cols = st.columns(3)
    relevance = event.get('relevance', 'Low')
    signal = event.get('signal_strength', 'Weak Signal')
    category = event.get('category', 'Unknown')
    
    cols[0].markdown(get_badge_html('Relevance', relevance, 'relevance'), unsafe_allow_html=True)
    cols[1].markdown(get_badge_html('Category', category, 'category').replace('metadata-badge', 'metadata-badge signal-weak'), unsafe_allow_html=True) # Use a generic color for category
    cols[2].markdown(get_badge_html('Signal', signal, 'signal'), unsafe_allow_html=True)
    
    # Details
    render_event_details(event)

# --- Main App Logic ---

# Initialize session state for first-time logic
if 'cleared_stale_status' not in st.session_state:
    # Clear stale 'starting' status check (retained logic)
    status_file = Path('research_status.json')
    if status_file.exists():
        try:
            status_data = json.loads(status_file.read_text())
            if status_data.get('status') == 'starting':
                status_file.unlink()
        except: pass
    st.session_state['cleared_stale_status'] = True

# Sidebar (Refactored to functions where possible)
# Assuming these imports are at the top of your main file:
# import streamlit as st
# from datetime import datetime
# from pathlib import Path
# import time
# from your_module import get_all_research_files, parse_timestamp, check_research_status, run_research_backend

def display_log_with_wrapping(log_file_path: str):
    """Displays the last 30 lines of a log file with forced text wrapping."""
    log_path = Path(log_file_path)
    if log_path.exists():
        st.caption(f"📊 Log size: {log_path.stat().st_size:,} bytes")
        
        # Use a simple text area instead of expander for better sidebar compatibility
        # Reverting to expander to maintain original UI structure, but use the wrapping fix.
        with st.expander("📋 View Log (last 30 lines)", expanded=False):
            try:
                # 1. Read the last 30 lines of the log file
                log_content = log_path.read_text(encoding='utf-8', errors='ignore').splitlines()[-30:]
                
                # 2. Use st.markdown with CSS to force text wrapping
                styled_log = f"""
                    <div style="
                        border: 1px solid #333;
                        padding: 10px;
                        background-color: #0e1117; 
                        font-family: monospace;
                        white-space: pre-wrap; /* FORCES WRAPPING */
                        max-height: 250px; /* Adjusted height for sidebar */
                        overflow-y: auto;
                    ">
                        {'<br>'.join(log_content)}
                    </div>
                """
                st.markdown(styled_log, unsafe_allow_html=True)
                
            except Exception as e: 
                st.error(f"Cannot read log: {e}")


# Sidebar (Refactored to functions where possible)
with st.sidebar:
    st.markdown("## 🔍 CPF Research Intelligence")
    st.markdown("---")
    
    # Cloud Storage Status
    if CLOUD_ENABLED:
        try:
            if sheets_storage.initialize():
                st.success("☁️ Google Sheets: Connected")
            else:
                st.warning("☁️ Google Sheets: Not configured")
        except Exception as e:
            st.error(f"☁️ Google Sheets Error: {e}")
    else:
        st.info("☁️ Cloud storage: Disabled (local mode)")
    
    st.markdown("---")
    
    # Report Selection
    st.markdown("### 📊 Select Report")
    research_files = get_all_research_files()
    file_options = {}
    selected_file = None # Initialize selected_file
    
    if research_files:
        for f in research_files:
            dt = parse_timestamp(f)
            friendly_name = f"📄 {dt.strftime('%Y-%m-%d %H:%M:%S')}" if dt else f
            file_options[friendly_name] = f
        
        selected_friendly = st.selectbox("Choose a report:", options=list(file_options.keys()), index=0)
        selected_file = file_options[selected_friendly]
    else:
        st.warning("No research files found. Run a research job first!")
        
    st.markdown("---")
    
    # Run New Research
    st.markdown("### 🚀 Run New Research")
    status = check_research_status() # Check status first
    
    if status['status'] == 'running':
        st.markdown('<div class="running-indicator"><strong>⏳ Research Running...</strong><br>This may take 5-15 minutes.</div>', unsafe_allow_html=True)
        
        # Display elapsed time (retained logic)
        try:
            start_time = datetime.fromisoformat(status['start_time'])
            elapsed = datetime.now() - start_time
            st.info(f"⏱️ Running for: {int(elapsed.total_seconds() / 60)}m {int(elapsed.total_seconds() % 60)}s")
        except: 
            pass

        # Display PID and log (using the new function for log)
        if status.get('pid'): 
            st.caption(f"🔢 Process ID: {status['pid']}")
            
        if status.get('log_file'):
            display_log_with_wrapping(status['log_file']) 
            
        if st.button("🔄 Refresh Status"): 
            st.rerun()

    elif status['status'] == 'completed':
        st.success("✅ Research completed!")
        if st.button("🔄 Reload Reports"):
            Path('research_status.json').unlink(missing_ok=True)
            st.rerun()
            
    elif status['status'] == 'error':
        st.error(f"❌ Error: {status.get('error', 'Unknown error')}")
        if st.button("🔄 Clear Error"):
            Path('research_status.json').unlink(missing_ok=True)
            st.rerun()
            
    else:
        if st.button("▶️ Start Research Job", type="primary"):
            run_research_backend()
            st.success("✅ Research job started!")
            time.sleep(1)
            st.rerun()

# Main Content
if selected_file:
    data = load_research_data(selected_file)
    if data:
        st.markdown(f'<div class="main-header">{data.get("topic", "CPF Research Report")}</div>', unsafe_allow_html=True)
        
        # Metrics
        cols = st.columns(4)
        ts = parse_timestamp(selected_file)
        cols[0].metric("📅 Generated", ts.strftime('%Y-%m-%d') if ts else "N/A")
        cols[1].metric("📊 Total Events", len(data.get('events', [])))
        cols[2].metric("✅ Action Items", len(data.get('action_items', [])))
        cols[3].metric("📚 Sources", len(data.get('source', [])))
        
        st.markdown("---")
        
        # Executive Summary & Repeated Events
        with st.expander("📋 Executive Summary", expanded=True):
            st.markdown(data.get('summary', 'No summary available'))
        
        repeated_events = data.get('repeated_events', [])
        if repeated_events:
            with st.expander("🔁 Repeated Topics from Previous Reports", expanded=False):
                st.markdown("These topics have appeared in recent reports:")
                for summary in repeated_events:
                    # Extract the event title and occurrence count
                    title_part = summary.split('(seen')[0].strip()
                    count_part = summary.split('(seen')[1].split(')')[0].strip() if '(seen' in summary else ''
                    
                    st.markdown(f"**{title_part}**")
                    st.caption(f"Seen {count_part})" if count_part else "")
                    
                    # Parse and display previous occurrences with dates
                    lines = summary.split('\n')
                    if len(lines) > 1:
                        st.markdown("**Previous Occurrences:**")
                        for line in lines[1:]:
                            if line.strip().startswith('[Prev'):
                                # Extract date and description
                                # Format: [Prev #1 - 2025-11-15] Description...
                                parts = line.split('] ', 1)
                                if len(parts) == 2:
                                    header = parts[0].replace('[Prev #', '').strip()
                                    description = parts[1].strip()
                                    
                                    # Extract occurrence number and date
                                    if ' - ' in header:
                                        occ_num, date = header.split(' - ', 1)
                                        st.markdown(f"**📅 {date}** (Occurrence #{occ_num})")
                                    else:
                                        st.markdown(f"**Occurrence #{header.split(' - ')[0]}**")
                                    
                                    st.markdown(f"> {description}")
                                    st.markdown("")
                    
                    st.markdown("---")
        
        # Filters
        st.markdown("### 🔍 Filter Events")
        events = data.get('events', [])
        
        col1, col2, col3 = st.columns(3)
        
        categories = sorted(list(set(e.get('category', 'Unknown') for e in events)))
        with col1:
            selected_categories = st.multiselect("Category", options=categories, default=categories)
        
        relevance_options = ['High', 'Medium', 'Low']
        with col2:
            selected_relevance = st.multiselect("Relevance", options=relevance_options, default=relevance_options)
        
        signal_options = ['Established', 'Emerging', 'Weak Signal']
        with col3:
            selected_signals = st.multiselect("Signal Strength", options=signal_options, default=signal_options)
        
        search_query = st.text_input("🔎 Search in events (title, description, impact):", "")
        st.markdown("---")
        
        # Filter Logic (Simplified using list comprehension where possible)
        search_lower = search_query.lower()
        
        def filter_event(event):
            relevance = event.get('relevance', 'Low').lower()
            signal = event.get('signal_strength', 'Weak Signal').lower()
            searchable_text = f"{event.get('event', '')} {event.get('description', '')} {event.get('impact', '')}".lower()
            
            return (
                event.get('category', 'Unknown') in selected_categories and
                any(r.lower() in relevance for r in selected_relevance) and
                any(s.lower() in signal for s in selected_signals) and
                (not search_query or search_lower in searchable_text)
            )

        filtered_events = [e for e in events if filter_event(e)]

        # Display Events
        st.markdown(f"### 📌 Events ({len(filtered_events)} of {len(events)})")
        
        if not filtered_events:
            st.info("No events match the selected filters.")
        else:
            for idx, event in enumerate(filtered_events, 1):
                render_event_card(idx, event) # Use the refactored rendering function
                st.markdown("---") # Add a separator

        # Action Items & Primary Sources
        if data.get('action_items'):
            with st.expander("✅ Recommended Action Items", expanded=True):
                for idx, item in enumerate(data['action_items'], 1): st.markdown(f"{idx}. {item}")
        
        if data.get('source'):
            with st.expander("📚 Primary Sources", expanded=False):
                for idx, source in enumerate(data['source'], 1):
                    source = source.strip()
                    st.markdown(f"{idx}. [{source}]({source})" if source.startswith('http') else f"{idx}. {source}")
else:
    # No file selected message
    st.markdown('<div class="main-header">🔍 CPF Research Intelligence Dashboard</div>', unsafe_allow_html=True)
    st.info("👈 Select a report from the sidebar to view, or start a new research job.")
    st.markdown("---")
    st.markdown("### 📊 About This Dashboard")
    st.markdown("""
    This dashboard allows you to:
    - **View Research Reports**: Browse through past research outputs with detailed event analysis
    - **Filter Events**: Filter by category, relevance, signal strength, and search keywords
    - **Run New Research**: Trigger the backend research engine to generate new insights
    """)