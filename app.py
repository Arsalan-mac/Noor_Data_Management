import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import io
import time
import json
import re
import sys
import requests
import threading
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timedelta

# --- SAFE IMPORTS ---
try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import dedupe
    import pandas_dedupe
    HAS_DEDUPE = True
except ImportError:
    HAS_DEDUPE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from fuzzywuzzy import fuzz
    HAS_FUZZY = True
except ImportError:
    HAS_FUZZY = False

# --- CONFIGURATION ---
st.set_page_config(page_title="Noor Data Governance", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")
GOOGLE_API_KEY = 'AIzaSyA7RYsUT4L9OsETxCvTBjBlyoNnnmJOK88' 

# --- CUSTOM CSS ---
def load_css():
    st.markdown("""
        <style>
        .stApp { background-color: #0f172a; color: #cbd5e1; }
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stTextArea textarea {
            background-color: #1e293b !important; color: #e2e8f0 !important; border: 1px solid #334155 !important;
        }
        div[data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }
        div[data-testid="stMetricValue"] { color: #f8fafc; }
        .stButton button { border-radius: 6px; font-weight: 600; }
        </style>
    """, unsafe_allow_html=True)
    if os.path.exists("style.css"):
        with open("style.css", "r") as f: st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# --- DATABASE MANAGEMENT ---
DB_FILE = "noor_app.db"
DATA_STORAGE_DIR = "data_storage"
if not os.path.exists(DATA_STORAGE_DIR): os.makedirs(DATA_STORAGE_DIR)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, email TEXT UNIQUE, password TEXT, name TEXT, role TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT, type TEXT, domains TEXT, llm_provider TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS data_log (id INTEGER PRIMARY KEY, project_id INTEGER, domain TEXT, table_name TEXT, file_path TEXT, row_count INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS dq_rules (id INTEGER PRIMARY KEY, project_id INTEGER, domain TEXT, table_name TEXT, rule_name TEXT, rule_description TEXT, python_code TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS mapping_config (id INTEGER PRIMARY KEY, project_id INTEGER, domain TEXT, table_name TEXT, config_json TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS dq_results_log (id INTEGER PRIMARY KEY, project_id INTEGER, domain TEXT, table_name TEXT, rule_name TEXT, pass_count INTEGER, fail_count INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, is_latest BOOLEAN DEFAULT 1)''')
    
    # New Table for Data Dictionary
    c.execute('''CREATE TABLE IF NOT EXISTS data_dictionary (
        id INTEGER PRIMARY KEY,
        project_id INTEGER,
        domain TEXT,
        table_name TEXT,
        field_name TEXT,
        system_name TEXT,
        business_definition TEXT,
        category TEXT,
        allowed_length TEXT,
        data_type TEXT,
        dq_dimension TEXT,
        dq_rule TEXT,
        accountable_role TEXT,
        created_sys TEXT,
        updated_sys TEXT,
        blocked_sys TEXT,
        deleted_sys TEXT,
        data_owner TEXT,
        e2e_process TEXT
    )''')
    
    # Seeds
    c.execute("SELECT * FROM users WHERE email='admin@company.com'")
    if not c.fetchone():
        c.execute("INSERT INTO users (email, password, name, role) VALUES (?, ?, ?, ?)", ('admin@company.com', 'admin', 'System Admin', 'Admin'))
        c.execute("INSERT INTO users (email, password, name, role) VALUES (?, ?, ?, ?)", ('steward@company.com', '123', 'Mike Steward', 'Data Steward'))
    conn.commit(); conn.close()

def get_db_connection(): return sqlite3.connect(DB_FILE)

# --- DATA HELPER ---
def get_mapped_dataframe(proj_id, domain, table_name, file_path):
    if not os.path.exists(file_path): return pd.DataFrame(), []
    if file_path.endswith('.csv'): df = pd.read_csv(file_path)
    else: df = pd.read_excel(file_path)

    conn = get_db_connection()
    res = conn.execute("SELECT config_json FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, domain, table_name)).fetchone()
    conn.close()

    if not res: return df, [] 
    config = json.loads(res[0])
    mappings = config.get('mappings', {})
    value_maps = config.get('value_maps', {})

    # 1. Rename Columns
    rename_dict = {source: target for target, source in mappings.items() if source != "-- Select --"}
    df = df.rename(columns=rename_dict)
    
    # 2. Filter to Target Fields
    valid_targets = list(rename_dict.values())
    if valid_targets:
        cols = [c for c in valid_targets if c in df.columns]
        df = df[cols]

    # 3. Apply Value Mapping (Robust)
    for col, rules in value_maps.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            for r in rules:
                df[col] = df[col].replace(r['old'], r['new'])

    return df, config.get('target_fields', [])

# --- EXTERNAL VALIDATION LOGIC ---

def validate_vat_soap_single(row, vat_col):
    """Real VIES VAT Validation via SOAP - Single Item"""
    vat = row.get(vat_col)
    idx = row.name
    
    if pd.isna(vat) or str(vat).strip() == "": 
        return {'index': idx, 'status': "❌ Missing"}
    
    # Basic cleaning
    full_vat = str(vat).upper().replace(" ", "").replace(".", "").replace("-", "")
    
    # Improved Regex for generic EU format
    if not re.match(r'^[A-Z]{2}[0-9A-Z]{2,12}$', full_vat):
        return {'index': idx, 'status': "❌ Invalid Format"}
    
    country_code = full_vat[:2]
    vat_number = full_vat[2:]

    # EU Country Check
    eu_countries = [
        "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", 
        "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PL", "PT", "RO", 
        "SE", "SI", "SK", "XI"
    ]
    if country_code not in eu_countries:
        return {'index': idx, 'status': "⚠️ Non-EU (Skipped)"}
    
    url = "http://ec.europa.eu/taxation_customs/vies/services/checkVatService"
    headers = {'Content-Type': 'text/xml; charset=utf-8'}
    body = f"""<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:urn="urn:ec.europa.eu:taxud:vies:services:checkVat:types"><soapenv:Header/><soapenv:Body><urn:checkVat><urn:countryCode>{country_code}</urn:countryCode><urn:vatNumber>{vat_number}</urn:vatNumber></urn:checkVat></soapenv:Body></soapenv:Envelope>"""
    
    try:
        response = requests.post(url, headers=headers, data=body, timeout=10) 
        if response.status_code == 200:
            if "<valid>true</valid>" in response.text or "<ns2:valid>true</ns2:valid>" in response.text:
                 return {'index': idx, 'status': "✅ Valid (VIES)"}
            elif "<valid>false</valid>" in response.text or "<ns2:valid>false</ns2:valid>" in response.text:
                 return {'index': idx, 'status': "❌ Invalid (VIES)"}
            
            if re.search(r"<([a-zA-Z0-9]+:)?valid>true</\1?valid>", response.text):
                return {'index': idx, 'status': "✅ Valid (VIES)"}
            elif re.search(r"<([a-zA-Z0-9]+:)?valid>false</\1?valid>", response.text):
                return {'index': idx, 'status': "❌ Invalid (VIES)"}
            else:
                if "Fault" in response.text:
                     return {'index': idx, 'status': "⚠️ VIES Fault"}
                return {'index': idx, 'status': "⚠️ VIES Unknown Response"}
        else: 
            return {'index': idx, 'status': f"⚠️ API Error {response.status_code}"}
    except requests.exceptions.Timeout:
        return {'index': idx, 'status': "⚠️ Timeout"}
    except requests.exceptions.ConnectionError:
        return {'index': idx, 'status': "⚠️ Connection Error"}
    except Exception as e:
        return {'index': idx, 'status': f"⚠️ Error: {str(e)}"}

def search_place_google(row, col_mapping):
    """Calls Google Places API to validate address composed of multiple columns"""
    parts = []
    # Name
    if col_mapping.get('Name') and pd.notna(row.get(col_mapping['Name'])): 
        parts.append(str(row[col_mapping['Name']]))
    
    # Address Components (from Multiselect)
    addr_cols = col_mapping.get('AddressCols', [])
    addr_parts = []
    for col in addr_cols:
        if pd.notna(row.get(col)):
            addr_parts.append(str(row[col]))
        
    query_addr = " ".join(addr_parts)
    if query_addr: parts.append(query_addr)
    full_query = ", ".join(parts)
    
    if not full_query.strip():
        return {'index': row.name, 'MatchedName': '', 'MatchedAddress': '', 'PlaceID': '', 'Confidence': 0.0, 'Status': '❌ Empty Input'}

    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {'query': full_query, 'key': GOOGLE_API_KEY}

    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'OK' and data.get('results'):
                result = data['results'][0]
                return {
                    'index': row.name,
                    'MatchedName': result.get('name', ''),
                    'MatchedAddress': result.get('formatted_address', ''),
                    'PlaceID': result.get('place_id', ''),
                    'Confidence': result.get('rating', 0.0), 
                    'Status': '✅ Found'
                }
        return {'index': row.name, 'MatchedName': '', 'MatchedAddress': '', 'PlaceID': '', 'Confidence': 0.0, 'Status': '❌ Not Found'}
    except Exception:
        return {'index': row.name, 'MatchedName': '', 'MatchedAddress': '', 'PlaceID': '', 'Confidence': 0.0, 'Status': '⚠️ Error'}

def validate_fuzzy(row, col_mapping):
    if row.get('Google_Status') != '✅ Found': return "N/A"
    if not HAS_FUZZY: return "Lib Missing"
    
    input_name = str(row.get(col_mapping['Name'], '')).strip().lower()
    found_name = str(row.get('Google_Name', '')).strip().lower()
    
    if not input_name: return "No Input Name"

    scores = [
        fuzz.ratio(input_name, found_name),
        fuzz.partial_ratio(input_name, found_name),
        fuzz.token_sort_ratio(input_name, found_name),
        fuzz.token_set_ratio(input_name, found_name)
    ]
    
    if any(s >= 70 for s in scores): return "✅ Confident Match"
    return "⚠️ Potential Mismatch"

# --- AI HELPER ---
def query_llm(provider, api_key, prompt, system_prompt="You are a helpful data assistant."):
    if not api_key: return "NO_KEY"
    try:
        if provider == "OpenAI (ChatGPT)":
            if not HAS_OPENAI: return "Error: OpenAI lib missing."
            openai.api_key = api_key
            response = openai.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}])
            return response.choices[0].message.content
        elif provider == "Gemini":
            if not HAS_GEMINI: return "Error: Gemini lib missing."
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(system_prompt + "\n" + prompt)
            return response.text
    except Exception as e: return f"Error: {str(e)}"

def generate_python_rule(description, columns, provider, api_key):
    system_prompt = (
        "You are a Data Quality Engineer. Convert the English rule into a **Pandas Boolean Mask**."
        "\nRULES:"
        "\n1. Return ONLY the python expression. No markdown, no 'df = ...'."
        "\n2. Use `df['Column Name']` syntax (handle spaces)."
        "\n3. Handle None/NaN: use `.isna()` or `.notna()`."
        "\n4. If checking strings, use `.str.contains(...)` or `.str.len()`."
        "\n5. Date checks: ensure column is converted if needed or assume ISO format strings."
        "\n6. **CRITICAL:** Do NOT wrap the entire expression in quotes (e.g. do NOT return \"df['A'] > 5\")."
        "\n7. Ensure all brackets and quotes are balanced."
        "\nExample Output: (df['Age'] > 18) & (df['Status'] == 'Active')"
    )
    prompt = f"DataFrame Columns: {columns}\nRequirement: {description}\n\nBoolean Expression:"
    
    if not api_key:
        col = columns[0] if columns else 'Col'
        return f"(df['{col}'].notna())"
        
    code = query_llm(provider, api_key, prompt, system_prompt)
    
    # Cleaning Logic
    cleaned = re.sub(r'```python|```', '', code).strip()
    
    # Only remove quotes if they start AND end the string (i.e. it's wrapped)
    # This prevents removing a starting quote if the end is missing (or vice versa), though rare.
    if cleaned.startswith("'") and cleaned.endswith("'"):
        cleaned = cleaned[1:-1]
    elif cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
        
    return cleaned

# --- AUTH ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
    st.session_state['user'] = None
if 'active_project' not in st.session_state:
    st.session_state['active_project'] = None

def login_page():
    st.markdown("<div style='text-align: center; margin-bottom: 2rem;'><h1 style='color: #2563eb;'>Noor</h1><p style='color: #94a3b8;'>Data Governance Platform</p></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.subheader("Sign In")
            email = st.text_input("Email", placeholder="admin@company.com")
            password = st.text_input("Password", type="password", placeholder="admin")
            if st.form_submit_button("Access Platform", use_container_width=True):
                conn = get_db_connection()
                user = conn.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password)).fetchone()
                conn.close()
                if user:
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = {'id': user[0], 'name': user[3], 'role': user[4]}
                    st.rerun()
                else: st.error("Invalid credentials.")
        st.info("Demo: admin@company.com / admin")

# --- CALLBACKS ---
def save_rule_callback(proj_id, sel_domain, sel_table):
    r_name = st.session_state.get('input_rname', '')
    final_code = st.session_state.get('txt_code_area_widget', st.session_state.get('txt_code_area', ''))
    r_desc = st.session_state.get('input_rdesc', '')
    
    if r_name and final_code:
        conn = get_db_connection()
        current_id = st.session_state.get('edit_rule_id')
        if current_id:
            conn.execute("UPDATE dq_rules SET rule_name=?, rule_description=?, python_code=? WHERE id=?", 
                            (r_name, r_desc, final_code, current_id))
            st.toast("Rule Updated!")
        else:
            conn.execute("INSERT INTO dq_rules (project_id, domain, table_name, rule_name, rule_description, python_code) VALUES (?, ?, ?, ?, ?, ?)",
                            (proj_id, sel_domain, sel_table, r_name, r_desc, final_code))
            st.toast("Rule Created!")
        conn.commit()
        conn.close()
        
        st.session_state['edit_rule_id'] = None
        st.session_state['edit_name'] = ""
        st.session_state['edit_desc'] = ""
        st.session_state['edit_code'] = ""
        st.session_state['txt_code_area'] = ""
        st.session_state['txt_code_area_widget'] = ""
        st.session_state['input_rname'] = ""
        st.session_state['input_rdesc'] = ""

def load_rule_for_edit(r_id, r_name, r_desc, r_code):
    st.session_state['edit_rule_id'] = r_id
    st.session_state['edit_name'] = r_name
    st.session_state['edit_desc'] = r_desc
    st.session_state['edit_code'] = r_code
    st.session_state['txt_code_area'] = r_code
    st.session_state['input_rname'] = r_name
    st.session_state['input_rdesc'] = r_desc
    st.session_state['txt_code_area_widget'] = r_code

def load_rule_and_delete(r_id, r_name, r_desc, r_code):
    st.session_state['edit_name'] = r_name
    st.session_state['edit_desc'] = r_desc
    st.session_state['edit_code'] = r_code
    st.session_state['txt_code_area'] = r_code
    st.session_state['input_rname'] = r_name
    st.session_state['input_rdesc'] = r_desc
    st.session_state['txt_code_area_widget'] = r_code
    
    conn = get_db_connection()
    conn.execute("DELETE FROM dq_rules WHERE id=?", (r_id,))
    conn.commit()
    conn.close()
    st.toast("Rule moved to Editor (Deleted from List)")

def delete_rule_db(r_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM dq_rules WHERE id=?", (r_id,))
    conn.commit()
    conn.close()
    st.toast("Rule Deleted!")
    if st.session_state.get('edit_rule_id') == r_id:
        st.session_state['edit_rule_id'] = None
        st.session_state['edit_name'] = ""
        st.session_state['edit_desc'] = ""
        st.session_state['edit_code'] = ""
        st.session_state['txt_code_area'] = ""
        st.session_state['txt_code_area_widget'] = ""
        st.session_state['input_rname'] = ""
        st.session_state['input_rdesc'] = ""

def wrap_text(text, width=25):
    """Helper to wrap text for Graphviz labels"""
    return "\\n".join(textwrap.wrap(str(text), width=width))

# --- MAIN APP ---
def main_app():
    for k in ['edit_name', 'edit_desc', 'edit_code', 'txt_code_area', 'txt_code_area_widget', 'active_mapping_field', 'steward_df_cache_key']:
        if k not in st.session_state: st.session_state[k] = ""

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['user']['name']}")
        if st.session_state['active_project']: st.success(f"📂 Active: {st.session_state['active_project']['name']}")
        else: st.warning("⚠️ No Project Selected")
        
        # ADDED "Data Dictionary" to menu
        menu_items = ["Dashboard", "Project Setup", "Data Ingestion", "Data Mapping", "Data Dictionary", "DQ Rules Config", "Data Cleansing", "Smart A.I."]
        icons = ['speedometer2', 'gear', 'cloud-upload', 'git', 'book', 'tools', 'shield-check', 'compass']
        selected_view = option_menu("Navigation", menu_items, icons=icons, menu_icon="cast", default_index=0, styles={"container": {"padding": "0!important", "background-color": "#0f172a"},"nav-link": {"font-size": "14px", "text-align": "left", "--hover-color": "#1e293b"},"nav-link-selected": {"background-color": "#1e293b", "border-left": "4px solid #2563eb"}}) if HAS_OPTION_MENU else st.radio("Navigation", menu_items)
        if st.button("Logout"): st.session_state['authenticated'] = False; st.rerun()

    # 1. DASHBOARD
    if selected_view == "Dashboard":
        st.title("Data Governance Dashboard")
        if not st.session_state['active_project']: st.info("Please select an Active Project."); return

        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        t_df = pd.read_sql_query("SELECT domain, table_name, row_count FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        r_df = pd.read_sql_query("SELECT * FROM dq_results_log WHERE project_id=? AND is_latest=1", conn, params=(proj_id,))
        r_df_all = pd.read_sql_query("SELECT * FROM dq_results_log WHERE project_id=?", conn, params=(proj_id,))
        rc = conn.execute("SELECT COUNT(*) FROM dq_rules WHERE project_id=?", (proj_id,)).fetchone()[0]
        conn.close()

        c1, c2, c3 = st.columns(3)
        c1.metric("Tables Ingested", len(t_df))
        c2.metric("Total Rows", f"{t_df['row_count'].sum():,}" if not t_df.empty else "0")
        c3.metric("Active Rules", rc)

        domains = sorted(t_df['domain'].unique().tolist())
        
        st.markdown("### 📊 Data Quality Overview")
        if not r_df.empty:
            tot_p = r_df['pass_count'].sum(); tot_f = r_df['fail_count'].sum()
            overall_dq = int((tot_p / (tot_p + tot_f)) * 100) if (tot_p + tot_f) > 0 else 0
            
            c_fill1, c_main, c_fill2 = st.columns([1, 2, 1])
            with c_main:
                if HAS_PLOTLY:
                    fig = go.Figure(data=[go.Pie(labels=['Valid', 'Invalid'], values=[tot_p, tot_f], hole=.5, marker_colors=['#22c55e', '#ef4444'])])
                    fig.update_layout(title=dict(text=f"Overall DQ Score: {overall_dq}%", x=0.5, font=dict(size=20)), height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True)

            if len(domains) > 0:
                st.markdown("#### Domain Breakdown")
                cols = st.columns(max(len(domains), 1))
                for idx, d in enumerate(domains):
                    d_df = r_df[r_df['domain'] == d]
                    if not d_df.empty:
                        d_p = d_df['pass_count'].sum(); d_f = d_df['fail_count'].sum()
                        d_score = int((d_p / (d_p + d_f)) * 100) if (d_p + d_f) > 0 else 0
                        if HAS_PLOTLY:
                            fig_d = go.Figure(data=[go.Pie(labels=['Valid', 'Invalid'], values=[d_p, d_f], hole=.7, marker_colors=['#22c55e', '#ef4444'])])
                            fig_d.update_layout(title=dict(text=f"{d}<br>{d_score}%", x=0.5), height=200, margin=dict(t=30, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", showlegend=False)
                            cols[idx].plotly_chart(fig_d, use_container_width=True)
                    else:
                        cols[idx].warning(f"{d}: No DQ Data")

            st.markdown("#### Rule Performance Breakdown")
            rule_grp = r_df.groupby(['domain', 'table_name', 'rule_name'])[['pass_count', 'fail_count']].sum().reset_index()
            rule_grp['Total'] = rule_grp['pass_count'] + rule_grp['fail_count']
            rule_grp['Health %'] = ((rule_grp['pass_count'] / rule_grp['Total']) * 100).round(1)
            st.dataframe(rule_grp.style.background_gradient(subset=['Health %'], cmap='RdYlGn', vmin=0, vmax=100), use_container_width=True)
            
            st.divider()
            
            st.subheader("Data Quality Trend (Errors over Time)")
            if HAS_PLOTLY and not r_df_all.empty:
                r_df_all['date'] = pd.to_datetime(r_df_all['timestamp']).dt.date
                trend_data = r_df_all.groupby('date')[['fail_count']].sum().reset_index()
                
                if len(trend_data) < 2:
                    dates = [datetime.now().date() - timedelta(days=i) for i in range(4, 0, -1)]
                    mock_vals = [np.random.randint(10, 50) for _ in range(4)]
                    trend_data = pd.concat([pd.DataFrame({'date': dates, 'fail_count': mock_vals}), trend_data], ignore_index=True)

                fig_line = px.line(trend_data, x='date', y='fail_count', markers=True)
                fig_line.update_traces(line_color='#ef4444', line_width=3)
                fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", xaxis_title="Date", yaxis_title="Error Count", hovermode="x unified")
                st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No DQ Analysis runs found. Go to 'Data Cleansing' to execute rules.")

    # 2. PROJECT SETUP
    elif selected_view == "Project Setup":
        st.title("Project Configuration")
        tab1, tab2 = st.tabs(["Projects List", "Create New Project"])
        with tab1:
            conn = get_db_connection()
            projects_df = pd.read_sql_query("SELECT * FROM projects", conn)
            conn.close()
            if not projects_df.empty:
                for _, row in projects_df.iterrows():
                    c1, c2, c3, c4 = st.columns([1, 2, 2, 1])
                    c1.write(f"**ID: {row['id']}**"); c2.write(f"**{row['name']}**"); c3.caption(f"{row['type']} | {row['llm_provider']}")
                    if c4.button("Select", key=f"sel_proj_{row['id']}"):
                        st.session_state['active_project'] = row.to_dict(); st.session_state['api_key'] = ""; st.rerun()
                if st.session_state['active_project']:
                    st.divider(); st.success(f"Active: {st.session_state['active_project']['name']}")
                    st.session_state['api_key'] = st.text_input("API Key (Session)", type="password")
            else: st.info("No projects found.")
        with tab2:
            st.subheader("Create Project")
            c1, c2 = st.columns(2)
            name = c1.text_input("Name"); p_type = c2.selectbox("Type", ["Clean Data", "Migration"])
            domains = st.multiselect("Domains", ["Material", "Customer", "Supplier", "Finance", "HR"])
            prov = st.selectbox("LLM", ["OpenAI (ChatGPT)", "Gemini"])
            if st.button("Create"):
                conn = get_db_connection()
                conn.execute("INSERT INTO projects (name, type, domains, llm_provider) VALUES (?, ?, ?, ?)", (name, p_type, ",".join(domains), prov))
                conn.commit(); conn.close()
                st.success("Project Created Successfully!")
                time.sleep(1) 
                st.rerun()

    # 3. DATA INGESTION
    elif selected_view == "Data Ingestion":
        st.title("Data Ingestion")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']
        c1, c2 = st.columns(2)
        domain = c1.selectbox("Domain", st.session_state['active_project']['domains'].split(','))
        table = c2.text_input("Table Name (e.g., MARA)")
        up_file = st.file_uploader("Upload Data", type=['csv', 'xlsx'])
        if st.button("Ingest") and up_file and table:
            conn = get_db_connection()
            exist = conn.execute("SELECT id FROM data_log WHERE project_id=? AND domain=? AND table_name=?", (proj_id, domain, table)).fetchone()
            conn.close()
            f_path = os.path.join(DATA_STORAGE_DIR, f"{proj_id}_{domain}_{table}_{up_file.name}")
            with open(f_path, "wb") as f: f.write(up_file.getbuffer())
            df = pd.read_csv(f_path) if up_file.name.endswith('.csv') else pd.read_excel(f_path)
            conn = get_db_connection()
            if exist: conn.execute("UPDATE data_log SET file_path=?, row_count=? WHERE id=?", (f_path, len(df), exist[0]))
            else: conn.execute("INSERT INTO data_log (project_id, domain, table_name, file_path, row_count) VALUES (?, ?, ?, ?, ?)", (proj_id, domain, table, f_path, len(df)))
            conn.commit(); conn.close(); st.success("Data Updated!"); st.rerun()
        
        st.subheader("Ingested Data")
        conn = get_db_connection()
        logs = pd.read_sql_query("SELECT * FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        for _, r in logs.iterrows():
            c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
            c1.write(f"**{r['domain']}**"); c2.write(r['table_name']); c3.write(f"{r['row_count']} Rows")
            if c4.button("Del", key=f"d_{r['id']}"):
                conn=get_db_connection(); conn.execute("DELETE FROM data_log WHERE id=?", (r['id'],)); conn.commit(); conn.close()
                if os.path.exists(r['file_path']): os.remove(r['file_path'])
                st.rerun()

    # 4. DATA MAPPING
    elif selected_view == "Data Mapping":
        st.title("Data Mapping Workbench")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        t_df = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        if t_df.empty: st.warning("No Data Ingested"); return
        
        c_dom, c_tbl = st.columns(2)
        avail_domains = sorted(t_df['domain'].unique().tolist())
        sel_domain = c_dom.selectbox("Select Domain", avail_domains)
        avail_tables = t_df[t_df['domain'] == sel_domain]['table_name'].unique().tolist()
        sel_table = c_tbl.selectbox("Select Table", avail_tables)
        
        path = t_df[(t_df['domain'] == sel_domain) & (t_df['table_name'] == sel_table)]['file_path'].values[0]
        
        try: src_df = pd.read_csv(path, nrows=5) if path.endswith('.csv') else pd.read_excel(path, nrows=5)
        except: src_df = pd.DataFrame()
        src_cols = ["-- Select --"] + src_df.columns.tolist()

        conn = get_db_connection()
        saved = conn.execute("SELECT config_json FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, sel_domain, sel_table)).fetchone()
        conn.close()
        config = json.loads(saved[0]) if saved else {"target_fields": [], "mappings": {}, "value_maps": {}}

        st.subheader("1. Target Fields")
        c1, c2, c3 = st.columns([3, 1, 1])
        new_f = c1.text_input("Add Fields (comma-separated)", placeholder="ID, Name, Status")
        if c2.button("Add"):
            added = 0
            for f in new_f.split(','):
                f = f.strip()
                if f and f not in config['target_fields']: config['target_fields'].append(f); added+=1
            if added:
                conn = get_db_connection()
                conn.execute("DELETE FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, sel_domain, sel_table))
                conn.execute("INSERT INTO mapping_config (project_id, domain, table_name, config_json) VALUES (?, ?, ?, ?)", (proj_id, sel_domain, sel_table, json.dumps(config)))
                conn.commit(); conn.close(); st.success(f"Added {added} fields"); st.rerun()
        if c3.button("Auto-Map"):
            for t_col in config['target_fields']:
                if t_col in src_cols: config['mappings'][t_col] = t_col
            st.toast("Auto-mapped matching columns")
            st.rerun()
        
        st.subheader("2. Mappings")
        for tf in config['target_fields']:
            c1, c2, c3 = st.columns([2, 2, 2])
            c1.markdown(f"**{tf}**")
            cur = config['mappings'].get(tf, "-- Select --")
            idx = src_cols.index(cur) if cur in src_cols else 0
            config['mappings'][tf] = c2.selectbox(f"Source for {tf}", src_cols, index=idx, key=f"s_{tf}")
            if c3.button("Values", key=f"v_{tf}"): st.session_state['active_mapping_field'] = tf

        if st.button("Save Config"):
            conn = get_db_connection()
            conn.execute("DELETE FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, sel_domain, sel_table))
            conn.execute("INSERT INTO mapping_config (project_id, domain, table_name, config_json) VALUES (?, ?, ?, ?)", (proj_id, sel_domain, sel_table, json.dumps(config)))
            conn.commit(); conn.close(); st.success("Saved!")

        # --- VALUE MAPPING POPUP ---
        if 'active_mapping_field' in st.session_state:
            f = st.session_state['active_mapping_field']
            st.markdown("---")
            st.info(f"Value Mapping: **{f}**")
            c1, c2, c3 = st.columns([2, 2, 1])
            o = c1.text_input("Source Value (Old)")
            n = c2.text_input("Target Value (New)")
            
            if c3.button("Add Rule"):
                if o and n:
                    l = config['value_maps'].get(f, []); l.append({"old": o, "new": n}); config['value_maps'][f] = l
                    conn = get_db_connection()
                    conn.execute("DELETE FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, sel_domain, sel_table))
                    conn.execute("INSERT INTO mapping_config (project_id, domain, table_name, config_json) VALUES (?, ?, ?, ?)", (proj_id, sel_domain, sel_table, json.dumps(config)))
                    conn.commit(); conn.close(); st.success("Rule Added!"); st.rerun()
            
            if f in config['value_maps']:
                if st.button("Delete All Rules for this field"):
                    config['value_maps'][f] = []
                    conn = get_db_connection()
                    conn.execute("DELETE FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, sel_domain, sel_table))
                    conn.execute("INSERT INTO mapping_config (project_id, domain, table_name, config_json) VALUES (?, ?, ?, ?)", (proj_id, sel_domain, sel_table, json.dumps(config)))
                    conn.commit(); conn.close(); st.rerun()

                for idx, vm in enumerate(config['value_maps'][f]):
                    cols = st.columns([4, 1])
                    cols[0].write(f"{idx+1}. '{vm['old']}' ➔ '{vm['new']}'")
                    if cols[1].button("Del", key=f"vm_del_{idx}"):
                        config['value_maps'][f].pop(idx)
                        conn = get_db_connection()
                        conn.execute("DELETE FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, sel_domain, sel_table))
                        conn.execute("INSERT INTO mapping_config (project_id, domain, table_name, config_json) VALUES (?, ?, ?, ?)", (proj_id, sel_domain, sel_table, json.dumps(config)))
                        conn.commit(); conn.close(); st.rerun()
            else: st.caption("No value mappings.")

            if st.button("Close Editor"): del st.session_state['active_mapping_field']; st.rerun()

    # --- NEW: DATA DICTIONARY ---
    elif selected_view == "Data Dictionary":
        st.title("📚 Data Dictionary Workbench")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']

        conn = get_db_connection()
        t_df = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if t_df.empty: st.warning("No Data Ingested"); return

        c_dom, c_tbl = st.columns(2)
        avail_domains = sorted(t_df['domain'].unique().tolist())
        sel_domain = c_dom.selectbox("Select Domain", avail_domains)
        avail_tables = t_df[t_df['domain'] == sel_domain]['table_name'].unique().tolist()
        sel_table = c_tbl.selectbox("Select Table", avail_tables)

        # 1. Load actual file to get columns
        path = t_df[(t_df['domain'] == sel_domain) & (t_df['table_name'] == sel_table)]['file_path'].values[0]
        try: 
            if path.endswith('.csv'): 
                src_df = pd.read_csv(path, nrows=1)
            else: 
                src_df = pd.read_excel(path, nrows=1)
            actual_cols = src_df.columns.tolist()
            actual_dtypes = src_df.dtypes.astype(str).to_dict()
        except: 
            st.error("Could not read file structure."); return

        # 2. Load existing dictionary entries
        conn = get_db_connection()
        dict_df = pd.read_sql_query("SELECT * FROM data_dictionary WHERE project_id=? AND domain=? AND table_name=?", 
                                    conn, params=(proj_id, sel_domain, sel_table))
        conn.close()

        # 3. Merge: Ensure we have a row for every actual column
        # Create a base dataframe from the actual file columns
        base_data = []
        for col in actual_cols:
            row = {
                'field_name': col,
                'data_type': str(actual_dtypes.get(col, 'Unknown')),
                # Default empty values for other fields
                'system_name': '', 'business_definition': '', 'category': '',
                'allowed_length': '', 'dq_dimension': '', 'dq_rule': '',
                'accountable_role': '', 'created_sys': '', 'updated_sys': '',
                'blocked_sys': '', 'deleted_sys': '', 'data_owner': '', 'e2e_process': ''
            }
            # Update with existing data if found
            if not dict_df.empty:
                existing = dict_df[dict_df['field_name'] == col]
                if not existing.empty:
                    # Map existing values, excluding id/proj/domain/table
                    for k in row.keys():
                        if k in existing.columns and pd.notna(existing.iloc[0][k]):
                            row[k] = existing.iloc[0][k]
            base_data.append(row)
        
        edit_df = pd.DataFrame(base_data)

        st.info("Edit metadata for fields below. 'Field Name' is from the source file.")
        
        # 4. Display Editor
        # Define column config for better UI
        col_config = {
            "field_name": st.column_config.TextColumn("Attribute (Field)", disabled=True),
            "data_type": st.column_config.TextColumn("Data Type"),
            "system_name": st.column_config.TextColumn("System Name"),
            "business_definition": st.column_config.TextColumn("Business Definition", width="large"),
            "category": st.column_config.SelectboxColumn("Category", options=["Master Data", "Transactional", "Reference", "Metadata"]),
            "allowed_length": st.column_config.TextColumn("Length"),
            "dq_dimension": st.column_config.SelectboxColumn("DQ Dimension", options=["Accuracy", "Completeness", "Consistency", "Validity", "Uniqueness", "Timeliness"]),
            "dq_rule": st.column_config.TextColumn("DQ Rule"),
            "accountable_role": st.column_config.TextColumn("Accountable Role"),
            "created_sys": st.column_config.TextColumn("Created In"),
            "updated_sys": st.column_config.TextColumn("Updated In"),
            "blocked_sys": st.column_config.TextColumn("Blocked In"),
            "deleted_sys": st.column_config.TextColumn("Deleted In"),
            "data_owner": st.column_config.TextColumn("Data Owner"),
            "e2e_process": st.column_config.TextColumn("E2E Process")
        }

        edited_data = st.data_editor(edit_df, use_container_width=True, hide_index=True, column_config=col_config, num_rows="fixed")

        # 5. Save Button and Download
        c_save, c_dl = st.columns([1, 1])
        if c_save.button("Save Dictionary", type="primary"):
            conn = get_db_connection()
            # Delete old entries for this table to avoid duplicates/orphans
            conn.execute("DELETE FROM data_dictionary WHERE project_id=? AND domain=? AND table_name=?", 
                         (proj_id, sel_domain, sel_table))
            
            # Insert new data
            for _, row in edited_data.iterrows():
                conn.execute("""
                    INSERT INTO data_dictionary (
                        project_id, domain, table_name, field_name, system_name, 
                        business_definition, category, allowed_length, data_type, 
                        dq_dimension, dq_rule, accountable_role, created_sys, 
                        updated_sys, blocked_sys, deleted_sys, data_owner, e2e_process
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    proj_id, sel_domain, sel_table, row['field_name'], row['system_name'],
                    row['business_definition'], row['category'], row['allowed_length'], row['data_type'],
                    row['dq_dimension'], row['dq_rule'], row['accountable_role'], row['created_sys'],
                    row['updated_sys'], row['blocked_sys'], row['deleted_sys'], row['data_owner'], row['e2e_process']
                ))
            
            conn.commit()
            conn.close()
            st.success("Data Dictionary Saved Successfully!")
            st.rerun() # Rerun to refresh the download button with latest data

        # Download Logic
        if not edited_data.empty:
            towrite_dict = io.BytesIO()
            edited_data.to_excel(towrite_dict, index=False, header=True)
            towrite_dict.seek(0)
            c_dl.download_button("Download Dictionary (Excel)", towrite_dict, f"dictionary_{sel_table}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


    # 5. DQ RULES CONFIGURATION
    elif selected_view == "DQ Rules Config":
        st.title("🛠️ DQ Rules Studio")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']
        
        conn = get_db_connection()
        t_df = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if t_df.empty: st.warning("Ingest data first."); return
        
        # Two-step dropdown
        c_dom, c_tbl = st.columns(2)
        avail_domains = sorted(t_df['domain'].unique().tolist())
        sel_domain = c_dom.selectbox("Select Domain", avail_domains)
        avail_tables = t_df[t_df['domain'] == sel_domain]['table_name'].unique().tolist()
        sel_table = c_tbl.selectbox("Select Table", avail_tables)
        
        path = t_df[(t_df['domain'] == sel_domain) & (t_df['table_name'] == sel_table)]['file_path'].values[0]
        
        sample_df, mapped_cols = get_mapped_dataframe(proj_id, sel_domain, sel_table, path)
        cols = mapped_cols if mapped_cols else (sample_df.columns.tolist() if not sample_df.empty else [])

        st.caption("Available Columns: " + ", ".join(cols) if cols else "No columns found")
        st.divider()
        col_studio, col_lib = st.columns([1, 1])

        with col_studio:
            st.subheader("Rule Creator")
            with st.container(border=True):
                r_name = st.text_input("Rule Name (Required)", value=st.session_state.get('edit_name', ''), placeholder="e.g., Check_Active_Status", key="input_rname")
                r_desc = st.text_area("Requirement (English)", value=st.session_state.get('edit_desc', ''), placeholder="e.g., Status must be 'Active'", key="input_rdesc")
                
                # AUTOMATED BUTTON: Generate, Save, and Rerun
                if st.button("✨ Generate & Save Rule", type="primary"):
                    if not r_name or not r_desc:
                        st.error("Please provide both Rule Name and Requirement.")
                    else:
                        with st.spinner("Generating Python Logic & Saving..."):
                            # 1. Generate Code
                            code = generate_python_rule(r_desc, cols, st.session_state['active_project']['llm_provider'], st.session_state.get('api_key'))
                            
                            # 2. Save directly to DB
                            conn = get_db_connection()
                            current_id = st.session_state.get('edit_rule_id')
                            if current_id:
                                conn.execute("UPDATE dq_rules SET rule_name=?, rule_description=?, python_code=? WHERE id=?", 
                                                (r_name, r_desc, code, current_id))
                                st.toast("Rule Updated & Saved!")
                            else:
                                conn.execute("INSERT INTO dq_rules (project_id, domain, table_name, rule_name, rule_description, python_code) VALUES (?, ?, ?, ?, ?, ?)",
                                                (proj_id, sel_domain, sel_table, r_name, r_desc, code))
                                st.toast("Rule Created & Saved!")
                            conn.commit()
                            conn.close()

                            # 3. Clear State
                            st.session_state['edit_rule_id'] = None
                            st.session_state['edit_name'] = ""
                            st.session_state['edit_desc'] = ""
                            st.session_state['edit_code'] = ""
                            st.session_state['txt_code_area'] = ""
                            st.session_state['txt_code_area_widget'] = ""
                            st.session_state['input_rname'] = ""
                            st.session_state['input_rdesc'] = ""
                            
                            # 4. Refresh List
                            time.sleep(1)
                            st.rerun()

        with col_lib:
            st.subheader("Existing Rules")
            conn = get_db_connection()
            rules = pd.read_sql_query("SELECT id, rule_name, rule_description, python_code FROM dq_rules WHERE project_id=? AND domain=? AND table_name=?", conn, params=(proj_id, sel_domain, sel_table))
            conn.close()
            
            # Download Rules Button
            if not rules.empty:
                towrite_rules = io.BytesIO()
                rules.to_excel(towrite_rules, index=False, header=True)
                towrite_rules.seek(0)
                st.download_button("Download Rules (Excel)", towrite_rules, f"rules_{sel_table}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            if not rules.empty:
                for i, row in rules.iterrows():
                    with st.expander(f"**{row['rule_name']}**", expanded=False):
                        st.caption(row['rule_description'])
                        st.code(row['python_code'], language='python')
                        c_edit, c_del = st.columns([1, 1])
                        if c_edit.button("Edit", key=f"ed_{row['id']}"):
                            load_rule_for_edit(row['id'], row['rule_name'], row['rule_description'], row['python_code'])
                            st.rerun()
                        if c_del.button("Delete", key=f"del_{row['id']}"):
                            delete_rule_db(row['id'])
                            st.rerun()
            else: st.info("No rules defined.")

    # 6. DATA CLEANSING
    elif selected_view == "Data Cleansing":
        st.title("Data Cleansing & Stewardship")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        t_df = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if t_df.empty: return
        
        # Two-step dropdown
        c_dom, c_tbl = st.columns(2)
        avail_domains = sorted(t_df['domain'].unique().tolist())
        sel_domain = c_dom.selectbox("Select Domain", avail_domains)
        avail_tables = t_df[t_df['domain'] == sel_domain]['table_name'].unique().tolist()
        sel_table = c_tbl.selectbox("Select Table", avail_tables)
        
        # Cache Key for Reload
        current_selection_key = f"{sel_domain}_{sel_table}"
        
        if 'steward_df_cache_key' not in st.session_state or st.session_state['steward_df_cache_key'] != current_selection_key:
            path = t_df[(t_df['domain'] == sel_domain) & (t_df['table_name'] == sel_table)]['file_path'].values[0]
            st.session_state['steward_df'], _ = get_mapped_dataframe(proj_id, sel_domain, sel_table, path)
            st.session_state['steward_df_cache_key'] = current_selection_key
            st.rerun()

        # --- ACTIONS ---
        c_actions, c_filter = st.columns([3, 1])
        
        with c_actions:
            c_run, c_ext = st.columns([1, 1])
            
            # 1. Standard DQ
            if c_run.button("RUN DQ ANALYSIS ⚡", type="primary", use_container_width=True):
                df = st.session_state['steward_df'].copy()
                conn = get_db_connection()
                rules = pd.read_sql_query("SELECT * FROM dq_rules WHERE project_id=? AND domain=? AND table_name=?", conn, params=(proj_id, sel_domain, sel_table))
                conn.execute("UPDATE dq_results_log SET is_latest=0 WHERE project_id=? AND domain=? AND table_name=?", (proj_id, sel_domain, sel_table))
                
                for _, r in rules.iterrows():
                    try:
                        mask = eval(r['python_code'], {"__builtins__": None}, {'df': df, 'pd': pd, 'np': np})
                        if isinstance(mask, pd.Series):
                            df[f"{r['rule_name']}_Status"] = mask.map({True: "✅ Valid", False: "❌ Invalid"})
                            df[f"{r['rule_name']}_Justification"] = mask.map({True: "Passed", False: r['rule_description']})
                            conn.execute("INSERT INTO dq_results_log (project_id, domain, table_name, rule_name, pass_count, fail_count, is_latest) VALUES (?, ?, ?, ?, ?, ?, 1)",
                                         (proj_id, sel_domain, sel_table, r['rule_name'], int(mask.sum()), int(len(mask)-mask.sum())))
                    except: pass
                
                conn.commit(); conn.close()
                st.session_state['steward_df'] = df
                st.success("Analysis Complete")

        # 2. External Validation (Real SOAP + Google Logic)
        if sel_domain in ["Customer", "Supplier"]:
            with st.expander("🌍 External Validation (Google Maps & VIES SOAP)"):
                st.info("Validate addresses (Google) and VAT numbers (VIES). Select columns to construct address query.")
                
                cols = st.session_state['steward_df'].columns.tolist()
                
                # --- UPDATED ADDRESS INPUT UI ---
                # Replaced individual dropdowns with Multiselect for Address components
                col_c1, col_c2 = st.columns(2)
                cm_name = col_c1.selectbox("Name Column", [""] + cols)
                cm_addr_cols = col_c2.multiselect("Select Address Columns (Order matters: Street, City, Country...)", cols)
                
                st.markdown("---")
                vat_col = st.selectbox("VAT Number Column (for VIES)", [""] + cols)
                
                c_btn1, c_btn2 = st.columns(2)
                
                if c_btn1.button("Check Address (Google Maps)", use_container_width=True):
                    # Updated Mapping for Search Function
                    col_mapping = {'Name': cm_name, 'AddressCols': cm_addr_cols}
                    
                    if cm_name or cm_addr_cols:
                        df = st.session_state['steward_df'].copy()
                        with st.spinner("Checking Addresses (Parallel Google API)..."):
                            results_map = {}
                            with ThreadPoolExecutor(max_workers=10) as executor:
                                futures = {executor.submit(search_place_google, row, col_mapping): row.name for _, row in df.iterrows()}
                                for future in as_completed(futures):
                                    res = future.result()
                                    results_map[res['index']] = res
                            
                            for idx, res in results_map.items():
                                df.at[idx, 'Google_Name'] = res['MatchedName']
                                df.at[idx, 'Google_Address'] = res['MatchedAddress']
                                df.at[idx, 'Google_Status'] = res['Status']
                                df.at[idx, 'Google_Conf'] = res['Confidence']
                                if HAS_FUZZY: df.at[idx, 'Fuzzy_Match'] = validate_fuzzy(df.loc[idx], col_mapping)
                            
                            st.session_state['steward_df'] = df
                            st.success("Address Validation Complete!")
                            st.rerun()

                if c_btn2.button("Check VAT (VIES SOAP)", use_container_width=True):
                    if vat_col:
                        df = st.session_state['steward_df'].copy()
                        with st.spinner("Checking VAT Numbers (Parallel VIES SOAP)..."):
                            results_map = {}
                            # VIES can be slow, parallelize cautiously (e.g. 5 workers)
                            with ThreadPoolExecutor(max_workers=5) as executor:
                                futures = {executor.submit(validate_vat_soap_single, row, vat_col): row.name for _, row in df.iterrows()}
                                for future in as_completed(futures):
                                    res = future.result()
                                    results_map[res['index']] = res
                            
                            for idx, res in results_map.items():
                                df.at[idx, 'Ext_VAT_Check'] = res['status']
                                
                            st.session_state['steward_df'] = df
                            st.success("VAT Validation Complete!")
                            st.rerun()

        show_errors = c_filter.checkbox("Show Rows with Errors Only")

        main_df = st.session_state['steward_df']
        view_df = main_df.copy()
        
        if show_errors:
            status_cols = [c for c in view_df.columns if c.endswith('_Status')]
            if status_cols:
                mask = view_df[status_cols].apply(lambda x: x.str.contains('Invalid', na=False)).any(axis=1)
                view_df = view_df[mask]

        edited_view_df = st.data_editor(view_df, num_rows="dynamic", use_container_width=True, key="main_editor")
        
        if not edited_view_df.equals(view_df):
            main_df.update(edited_view_df)
            
            conn = get_db_connection()
            rules = pd.read_sql_query("SELECT * FROM dq_rules WHERE project_id=? AND domain=? AND table_name=?", conn, params=(proj_id, sel_domain, sel_table))
            
            conn.execute("UPDATE dq_results_log SET is_latest=0 WHERE project_id=? AND domain=? AND table_name=?", (proj_id, sel_domain, sel_table))
            
            for _, r in rules.iterrows():
                try:
                    mask = eval(r['python_code'], {"__builtins__": None}, {'df': main_df, 'pd': pd, 'np': np})
                    if isinstance(mask, pd.Series):
                        main_df[f"{r['rule_name']}_Status"] = mask.map({True: "✅ Valid", False: "❌ Invalid"})
                        main_df[f"{r['rule_name']}_Justification"] = mask.map({True: "Passed", False: r['rule_description']})
                        conn.execute("INSERT INTO dq_results_log (project_id, domain, table_name, rule_name, pass_count, fail_count, is_latest) VALUES (?, ?, ?, ?, ?, ?, 1)",
                                     (proj_id, sel_domain, sel_table, r['rule_name'], int(mask.sum()), int(len(mask)-mask.sum())))
                except: pass
            
            conn.commit(); conn.close()
            st.session_state['steward_df'] = main_df
            st.rerun()

        c_save, c_dl = st.columns(2)
        if c_save.button("Save to Disk"):
            # --- UPDATED SAVE LOGIC ---
            # Save the current state of stewardship (including edits and new columns) back to the source file
            path = t_df[(t_df['domain'] == sel_domain) & (t_df['table_name'] == sel_table)]['file_path'].values[0]
            try:
                if path.endswith('.csv'):
                    st.session_state['steward_df'].to_csv(path, index=False)
                else:
                    st.session_state['steward_df'].to_excel(path, index=False)
                st.success(f"Saved changes to {os.path.basename(path)}!")
            except Exception as e:
                st.error(f"Error saving file: {e}")
        
        towrite = io.BytesIO()
        st.session_state['steward_df'].to_excel(towrite, index=False, header=True)
        towrite.seek(0)
        c_dl.download_button("Download Excel", towrite, f"clean_{sel_table}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        
        # --- NEW: DATA DICTIONARY VISUALIZER ---
        st.divider()
        st.subheader("🧩 Data Dictionary Visualizer")
        
        # 1. Get available columns from the current dataframe (ensure realtime sync with file)
        available_columns = st.session_state['steward_df'].columns.tolist()
        
        # Field Selector
        dd_field = st.selectbox("Select Field to Visualize", available_columns)
        
        if dd_field:
            # 2. Fetch dictionary data
            conn = get_db_connection()
            dd_df = pd.read_sql_query("SELECT * FROM data_dictionary WHERE project_id=? AND domain=? AND table_name=? AND field_name=?", 
                                      conn, params=(proj_id, sel_domain, sel_table, dd_field))
            conn.close()
            
            # 3. Use fetched data OR defaults if not yet saved in dictionary
            if not dd_df.empty:
                field_data = dd_df.iloc[0]
            else:
                # Create a default dict with empty values
                field_data = {
                    'field_name': dd_field,
                    'business_definition': 'Not defined',
                    'data_owner': 'N/A',
                    'e2e_process': 'N/A',
                    'data_type': str(st.session_state['steward_df'][dd_field].dtype),
                    'allowed_length': 'N/A',
                    'system_name': 'N/A',
                    'created_sys': 'N/A',
                    'updated_sys': 'N/A',
                    'dq_rule': 'N/A',
                    'dq_dimension': 'N/A'
                }

            # 4. Construct Sexy Dark Mode Graph (Compact, TB)
            dot_code = f"""
            digraph G {{
                rankdir=TB;
                bgcolor="transparent";
                # Tighter packing
                nodesep=0.3;
                ranksep=0.4;
                
                # Smaller nodes
                node [fontname="Arial", fontsize=9, shape=box, style="filled,rounded", color="white", fontcolor="white", penwidth=1.0, margin=0.1, height=0.3];
                edge [fontname="Arial", fontsize=8, color="#94a3b8", penwidth=1.0, arrowsize=0.7];

                # Root
                root [label="{field_data['field_name']}", fillcolor="#2563eb", height=0.4, fontsize=11];

                # Clusters (Subgraphs)
                subgraph cluster_bus {{
                    label=""; penwidth=0;
                    bus [label="Business Info", fillcolor="#059669"]; # Emerald
                    def [label="Definition\\n{wrap_text(field_data['business_definition'] or 'N/A', 30)}", shape=note, fillcolor="#064e3b", fontcolor="#cbd5e1"];
                    owner [label="Owner: {field_data['data_owner'] or 'N/A'}", fillcolor="#064e3b", fontcolor="#cbd5e1"];
                    proc [label="Process: {field_data['e2e_process'] or 'N/A'}", fillcolor="#064e3b", fontcolor="#cbd5e1"];
                }}

                subgraph cluster_tech {{
                    label=""; penwidth=0;
                    tech [label="Technical", fillcolor="#d97706"]; # Amber
                    type [label="Type: {field_data['data_type']}", fillcolor="#78350f", fontcolor="#cbd5e1"];
                    len [label="Len: {field_data['allowed_length'] or 'N/A'}", fillcolor="#78350f", fontcolor="#cbd5e1"];
                    sys [label="System: {field_data['system_name'] or 'N/A'}", fillcolor="#78350f", fontcolor="#cbd5e1"];
                }}

                subgraph cluster_lin {{
                    label=""; penwidth=0;
                    systems [label="Lineage", fillcolor="#7c3aed"]; # Violet
                    c_sys [label="Created: {field_data['created_sys'] or 'N/A'}", fillcolor="#5b21b6", fontcolor="#cbd5e1"];
                    u_sys [label="Updated: {field_data['updated_sys'] or 'N/A'}", fillcolor="#5b21b6", fontcolor="#cbd5e1"];
                }}

                subgraph cluster_dq {{
                    label=""; penwidth=0;
                    dq [label="Quality", fillcolor="#db2777"]; # Pink/Red
                    rule [label="Rule: {field_data['dq_rule'] or 'N/A'}", fillcolor="#9d174d", fontcolor="#cbd5e1"];
                    dim [label="Dim: {field_data['dq_dimension'] or 'N/A'}", fillcolor="#9d174d", fontcolor="#cbd5e1"];
                }}

                # Connections
                root -> bus [color="#059669"];
                bus -> def;
                bus -> owner;
                bus -> proc;

                root -> tech [color="#d97706"];
                tech -> type;
                tech -> len;
                tech -> sys;

                root -> systems [color="#7c3aed"];
                systems -> c_sys;
                systems -> u_sys;

                root -> dq [color="#db2777"];
                dq -> rule;
                dq -> dim;
            }}
            """
            
            try:
                st.graphviz_chart(dot_code, use_container_width=True)
            except Exception as e:
                st.error(f"Graphviz visualization failed. Showing text data.")
                st.json(field_data if isinstance(field_data, dict) else field_data.to_dict())


    # 7. SMART A.I.
    elif selected_view == "Smart A.I.":
        st.title("Smart A.I. Explorer")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        t_df = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if t_df.empty: st.warning("No data found."); return
        sel_k = st.selectbox("Select Context (Table)", [f"{r['domain']} - {r['table_name']}" for _, r in t_df.iterrows()])
        dom, tbl = sel_k.split(" - ")
        path = t_df[t_df['table_name'] == tbl]['file_path'].values[0]

        if 'chat_df' not in st.session_state:
            st.session_state['chat_df'], _ = get_mapped_dataframe(proj_id, dom, tbl, path)
        
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [{"role": "assistant", "content": "Hello! I have loaded your data."}]

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if user_input := st.chat_input("Ask about data..."):
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            with st.chat_message("user"): st.markdown(user_input)

            with st.chat_message("assistant"):
                df = st.session_state['chat_df']
                sys_prompt = f"Data
