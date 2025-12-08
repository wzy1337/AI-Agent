# -*- coding: utf-8 -*-
"""
Cloud Storage Module using Google Sheets.

Simple cloud storage for research outputs using Google Sheets as the backend.
This is easy to set up, free, and provides a nice UI for viewing historical data.

Setup Instructions:
==================
1. Go to Google Cloud Console (https://console.cloud.google.com/)
2. Create a new project (or use existing)
3. Enable "Google Sheets API" and "Google Drive API"
4. Go to "Credentials" → "Create Credentials" → "Service Account"
5. Download the JSON key file
6. Create a Google Sheet and share it with the service account email (with Editor access)
7. Copy the Sheet ID from the URL (the long string between /d/ and /edit)

For Streamlit Cloud:
-------------------
Add to your Streamlit secrets (Settings → Secrets):

[gcp_service_account]
type = "service_account"
project_id = "your-project-id"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\\n...\\n-----END PRIVATE KEY-----\\n"
client_email = "your-service@your-project.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"

[google_sheets]
spreadsheet_id = "your-spreadsheet-id-from-url"
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional, Any
import streamlit as st

# Check if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_SHARING_MODE') or os.environ.get('STREAMLIT_RUNTIME_ENV')


class GoogleSheetsStorage:
    """
    Google Sheets storage backend for research outputs.
    
    Structure:
    - Sheet "Events": One row per event with all fields as columns
    - Sheet "Reports": Metadata about each research run
    """
    
    def __init__(self):
        self.client = None
        self.spreadsheet = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """Initialize Google Sheets connection."""
        if self._initialized:
            return True
            
        try:
            import gspread
            from google.oauth2.service_account import Credentials
            
            # Get credentials from Streamlit secrets
            if "gcp_service_account" not in st.secrets:
                st.warning("⚠️ Google Sheets not configured. Add 'gcp_service_account' to Streamlit secrets.")
                return False
            
            if "google_sheets" not in st.secrets or "spreadsheet_id" not in st.secrets["google_sheets"]:
                st.warning("⚠️ Spreadsheet ID not configured. Add 'google_sheets.spreadsheet_id' to secrets.")
                return False
            
            # Create credentials
            credentials_dict = dict(st.secrets["gcp_service_account"])
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            credentials = Credentials.from_service_account_info(credentials_dict, scopes=scopes)
            
            # Connect to Google Sheets
            self.client = gspread.authorize(credentials)
            spreadsheet_id = st.secrets["google_sheets"]["spreadsheet_id"]
            self.spreadsheet = self.client.open_by_key(spreadsheet_id)
            
            # Ensure required sheets exist
            self._ensure_sheets_exist()
            
            self._initialized = True
            return True
            
        except ImportError:
            st.warning("⚠️ gspread not installed. Run: pip install gspread google-auth")
            return False
        except Exception as e:
            st.error(f"❌ Failed to connect to Google Sheets: {e}")
            return False
    
    def _ensure_sheets_exist(self):
        """Create required sheets if they don't exist."""
        existing_sheets = [ws.title for ws in self.spreadsheet.worksheets()]
        
        # Events sheet - stores individual events
        if "Events" not in existing_sheets:
            events_sheet = self.spreadsheet.add_worksheet(title="Events", rows=1000, cols=20)
            events_sheet.update('A1:O1', [[
                'report_id', 'timestamp', 'event_name', 'category', 'description', 'relevance',
                'signal_strength', 'impact', 'scenario', 'policy_intervention',
                'location', 'actors', 'confidence', 'sources', 'dates'
            ]])
        
        # Reports sheet - stores report metadata
        if "Reports" not in existing_sheets:
            reports_sheet = self.spreadsheet.add_worksheet(title="Reports", rows=500, cols=10)
            reports_sheet.update('A1:G1', [[
                'report_id', 'timestamp', 'total_events', 'executive_summary',
                'repeated_events_count', 'topic', 'json_backup'
            ]])
    
    def save_research_output(self, data: Dict[str, Any], report_id: str = None) -> str:
        """
        Save research output to Google Sheets.
        
        Args:
            data: Research output dictionary (with 'events' key)
            report_id: Optional report ID (auto-generated if not provided)
            
        Returns:
            report_id
        """
        if not self.initialize():
            return None
            
        try:
            if report_id is None:
                report_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            timestamp = datetime.now().isoformat()
            
            # Get or extract events
            events = data.get('events', [])
            if not events and 'ResearchResponse' in data:
                events = data['ResearchResponse'].get('events', [])
            
            # Save to Events sheet
            events_sheet = self.spreadsheet.worksheet("Events")
            
            rows_to_add = []
            for event in events:
                row = [
                    report_id,
                    timestamp,
                    event.get('event', ''),
                    event.get('category', ''),
                    event.get('description', ''),
                    event.get('relevance', ''),
                    event.get('signal_strength', ''),
                    event.get('impact', ''),
                    event.get('scenario', ''),
                    event.get('policy_intervention', ''),
                    str(event.get('location', '')),
                    ', '.join(event.get('actors', [])) if isinstance(event.get('actors'), list) else str(event.get('actors', '')),
                    event.get('confidence', ''),
                    ', '.join(event.get('source', [])) if isinstance(event.get('source'), list) else str(event.get('source', '')),
                    ', '.join(event.get('date', [])) if isinstance(event.get('date'), list) else str(event.get('date', ''))
                ]
                rows_to_add.append(row)
            
            if rows_to_add:
                events_sheet.append_rows(rows_to_add, value_input_option='USER_ENTERED')
            
            # Save to Reports sheet
            reports_sheet = self.spreadsheet.worksheet("Reports")
            
            # Create a compact JSON backup (without full content to save space)
            json_backup = json.dumps({
                'event_count': len(events),
                'event_names': [e.get('event', '') for e in events],
                'repeated_events': data.get('repeated_events', [])
            }, ensure_ascii=False)
            
            # Get executive summary from data
            executive_summary = data.get('summary', '')
            if not executive_summary and 'ResearchResponse' in data:
                executive_summary = data['ResearchResponse'].get('summary', '')
            
            topic = data.get('topic', '')
            if not topic and 'ResearchResponse' in data:
                topic = data['ResearchResponse'].get('topic', '')
            
            report_row = [
                report_id,
                timestamp,
                len(events),
                executive_summary[:50000] if executive_summary else '',  # Limit to 50KB
                len(data.get('repeated_events', [])),
                topic,
                json_backup[:50000]  # Limit to 50KB per cell
            ]
            
            reports_sheet.append_row(report_row, value_input_option='USER_ENTERED')
            
            st.success(f"☁️ Saved {len(events)} events to Google Sheets (Report: {report_id})")
            return report_id
            
        except Exception as e:
            st.error(f"❌ Failed to save to Google Sheets: {e}")
            return None
    
    def get_all_reports(self, max_n: int = 20) -> List[Dict]:
        """Get list of all reports from Google Sheets."""
        if not self.initialize():
            return []
            
        try:
            reports_sheet = self.spreadsheet.worksheet("Reports")
            records = reports_sheet.get_all_records()
            
            # Sort by timestamp descending and limit
            records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return records[:max_n]
            
        except Exception as e:
            st.error(f"❌ Failed to get reports: {e}")
            return []
    
    def get_events_by_report(self, report_id: str) -> List[Dict]:
        """Get all events for a specific report."""
        if not self.initialize():
            return []
            
        try:
            events_sheet = self.spreadsheet.worksheet("Events")
            records = events_sheet.get_all_records()
            
            # Filter by report_id
            return [r for r in records if r.get('report_id') == report_id]
            
        except Exception as e:
            st.error(f"❌ Failed to get events: {e}")
            return []
    
    def get_all_events(self, max_n: int = 500) -> List[Dict]:
        """Get all events across all reports."""
        if not self.initialize():
            return []
            
        try:
            events_sheet = self.spreadsheet.worksheet("Events")
            records = events_sheet.get_all_records()
            
            # Sort by timestamp descending
            records.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return records[:max_n]
            
        except Exception as e:
            st.error(f"❌ Failed to get all events: {e}")
            return []
    
    def get_event_history(self, event_name: str) -> List[Dict]:
        """
        Get historical tracking of a specific event across reports.
        Useful for tracking how an emerging event develops over time.
        """
        if not self.initialize():
            return []
            
        try:
            events_sheet = self.spreadsheet.worksheet("Events")
            records = events_sheet.get_all_records()
            
            # Find events with similar names (fuzzy match)
            matching = []
            event_lower = event_name.lower()
            for r in records:
                if event_lower in r.get('event_name', '').lower():
                    matching.append(r)
            
            # Sort by timestamp
            matching.sort(key=lambda x: x.get('timestamp', ''))
            return matching
            
        except Exception as e:
            st.error(f"❌ Failed to get event history: {e}")
            return []
    
    def get_previous_event_names(self, max_reports: int = 2) -> set:
        """Get event names from recent reports for deduplication."""
        if not self.initialize():
            return set()
            
        try:
            reports = self.get_all_reports(max_n=max_reports)
            report_ids = [r.get('report_id') for r in reports]
            
            events_sheet = self.spreadsheet.worksheet("Events")
            records = events_sheet.get_all_records()
            
            event_names = set()
            for r in records:
                if r.get('report_id') in report_ids:
                    event_names.add(r.get('event_name', ''))
            
            return event_names
            
        except Exception as e:
            return set()


# Singleton instance
sheets_storage = GoogleSheetsStorage()


def save_to_cloud(data: Dict[str, Any], report_id: str = None) -> Optional[str]:
    """
    Save research output to Google Sheets (if configured).
    
    Call this after saving locally to sync to cloud.
    """
    if IS_STREAMLIT_CLOUD or _has_sheets_config():
        return sheets_storage.save_research_output(data, report_id)
    return None


def _has_sheets_config() -> bool:
    """Check if Google Sheets is configured in secrets."""
    try:
        return "gcp_service_account" in st.secrets and "google_sheets" in st.secrets
    except:
        return False


def get_cloud_reports(max_n: int = 20) -> List[Dict]:
    """Get all reports from cloud storage."""
    return sheets_storage.get_all_reports(max_n)


def get_cloud_events(report_id: str = None, max_n: int = 500) -> List[Dict]:
    """Get events from cloud storage."""
    if report_id:
        return sheets_storage.get_events_by_report(report_id)
    return sheets_storage.get_all_events(max_n)
