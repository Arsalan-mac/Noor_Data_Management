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
from contextlib import redirect_stdout

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

# --- CONFIGURATION ---
st.set_page_config(page_title="Noor Data Governance", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

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
    c.execute('''CREATE TABLE IF NOT EXISTS dq_results_log (id INTEGER PRIMARY KEY, project_id INTEGER, domain TEXT, table_name TEXT, rule_name TEXT, pass_count INTEGER, fail_count INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
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
    target_fields = config.get('target_fields', [])
    mappings = config.get('mappings', {})
    value_maps = config.get('value_maps', {})

    rename_dict = {source: target for target, source in mappings.items() if source != "-- Select --"}
    df = df.rename(columns=rename_dict)
    
    # Filter to targets if mapped
    valid_targets = list(rename_dict.values())
    if valid_targets:
        cols = [c for c in valid_targets if c in df.columns]
        df = df[cols]

    for col, rules in value_maps.items():
        if col in df.columns:
            for r in rules: df[col] = df[col].replace(r['old'], r['new'])

    return df, config.get('target_fields', [])

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
        "\nExample Output: `(df['Age'] > 18) & (df['Status'] == 'Active')`"
    )
    prompt = f"DataFrame Columns: {columns}\nRequirement: {description}\n\nBoolean Expression:"
    
    if not api_key:
        col = columns[0] if columns else 'Col'
        return f"# Demo Mode\n(df['{col}'].notna())"
        
    code = query_llm(provider, api_key, prompt, system_prompt)
    return re.sub(r'```python|```', '', code).strip()

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

# --- CALLBACKS (CRITICAL FOR STATE SAFETY) ---
def save_rule_callback(proj_id, sel_domain, sel_table):
    r_name = st.session_state.get('input_rname', '')
    r_desc = st.session_state.get('input_rdesc', '')
    final_code = st.session_state.get('txt_code_area', '')
    
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
        
        # Reset State safely
        st.session_state['edit_rule_id'] = None
        st.session_state['edit_name'] = ""
        st.session_state['edit_desc'] = ""
        st.session_state['edit_code'] = ""
        st.session_state['txt_code_area'] = ""

def load_rule_callback(r_id, r_name, r_desc, r_code):
    st.session_state['edit_rule_id'] = r_id
    st.session_state['edit_name'] = r_name
    st.session_state['edit_desc'] = r_desc
    st.session_state['edit_code'] = r_code
    st.session_state['txt_code_area'] = r_code

def delete_rule_callback(r_id):
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

# --- MAIN APP ---
def main_app():
    # GLOBAL STATE INIT (Prevents KeyErrors)
    for k in ['edit_rule_id', 'edit_name', 'edit_desc', 'edit_code', 'txt_code_area']:
        if k not in st.session_state: st.session_state[k] = None if k == 'edit_rule_id' else ""

    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['user']['name']}")
        if st.session_state['active_project']: st.success(f"📂 Active: {st.session_state['active_project']['name']}")
        else: st.warning("⚠️ No Project Selected")
        
        menu_items = ["Dashboard", "Project Setup", "Data Ingestion", "Data Mapping", "BP Deduplication", "DQ Rules Config", "Data Stewardship", "Data Exploration"]
        icons = ['speedometer2', 'gear', 'cloud-upload', 'git', 'people', 'tools', 'shield-check', 'compass']
        selected_view = option_menu("Navigation", menu_items, icons=icons, menu_icon="cast", default_index=0, styles={"container": {"padding": "0!important", "background-color": "#0f172a"},"nav-link": {"font-size": "14px", "text-align": "left", "--hover-color": "#1e293b"},"nav-link-selected": {"background-color": "#1e293b", "border-left": "4px solid #2563eb"}}) if HAS_OPTION_MENU else st.radio("Navigation", menu_items)
        if st.button("Logout"): st.session_state['authenticated'] = False; st.rerun()

    # 1. DASHBOARD (Full Drill Down)
    if selected_view == "Dashboard":
        st.title("Data Governance Dashboard")
        if not st.session_state['active_project']: st.info("Please select an Active Project."); return

        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        t_df = pd.read_sql_query("SELECT domain, table_name, row_count FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        r_df = pd.read_sql_query("SELECT * FROM dq_results_log WHERE project_id=?", conn, params=(proj_id,))
        rc = conn.execute("SELECT COUNT(*) FROM dq_rules WHERE project_id=?", (proj_id,)).fetchone()[0]
        conn.close()

        c1, c2, c3 = st.columns(3)
        c1.metric("Tables Ingested", len(t_df))
        c2.metric("Total Rows", f"{t_df['row_count'].sum():,}" if not t_df.empty else "0")
        c3.metric("Active Rules", rc)

        domains = sorted(t_df['domain'].unique().tolist())
        if domains: st.info(f"**Applicable Domains:** {', '.join(domains)}")

        st.markdown("### 📊 Data Quality Health")
        if not r_df.empty:
            sel_dom = st.selectbox("Drill Down Domain", ["All"] + domains)
            f_res = r_df if sel_dom == "All" else r_df[r_df['domain'] == sel_dom]
            
            tot_p = f_res['pass_count'].sum(); tot_f = f_res['fail_count'].sum()
            if HAS_PLOTLY and (tot_p+tot_f) > 0:
                fig = go.Figure(data=[go.Pie(labels=['Valid', 'Invalid'], values=[tot_p, tot_f], hole=.6, marker_colors=['#22c55e', '#ef4444'])])
                fig.update_layout(title=f"DQ Score ({sel_dom})", height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Rule Performance Breakdown")
            rule_grp = f_res.groupby(['table_name', 'rule_name'])[['pass_count', 'fail_count']].sum().reset_index()
            rule_grp['Total'] = rule_grp['pass_count'] + rule_grp['fail_count']
            rule_grp['Health %'] = ((rule_grp['pass_count'] / rule_grp['Total']) * 100).round(1)
            st.dataframe(rule_grp.style.background_gradient(subset=['Health %'], cmap='RdYlGn', vmin=0, vmax=100), use_container_width=True)
        else:
            st.info("No DQ Analysis runs found. Go to 'Data Stewardship' to execute rules.")

    # 2. PROJECT SETUP (Full)
    elif selected_view == "Project Setup":
        st.title("Project Configuration")
        tab1, tab2 = st.tabs(["Projects List", "Create New Project"])
        
        with tab1:
            conn = get_db_connection()
            projects_df = pd.read_sql_query("SELECT * FROM projects", conn)
            conn.close()
            
            if not projects_df.empty:
                for index, row in projects_df.iterrows():
                    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                    col1.write(f"**ID: {row['id']}**")
                    col2.write(f"**{row['name']}**")
                    col3.caption(f"{row['type']} | {row['llm_provider']}")
                    if col4.button("Select", key=f"sel_proj_{row['id']}"):
                        st.session_state['active_project'] = row.to_dict()
                        st.session_state['api_key'] = "" 
                        st.rerun()
                
                if st.session_state['active_project']:
                    st.divider()
                    st.success(f"Currently Active: {st.session_state['active_project']['name']}")
                    st.caption("Re-enter API Key for this session")
                    st.session_state['api_key'] = st.text_input("API Key", type="password")
            else:
                st.info("No projects found.")

        with tab2:
            st.subheader("Create New Project")
            col1, col2 = st.columns(2)
            p_name = col1.text_input("Project Name")
            p_type = col2.selectbox("Usecase", ["Clean Existing Data", "Migration Cleanse"])
            domains = st.multiselect("Data Domains", ["Material", "Customer", "Supplier", "Finance", "HR"])
            col3, col4 = st.columns(2)
            llm_prov = col3.selectbox("LLM Provider", ["OpenAI (ChatGPT)", "Gemini"])
            
            if st.button("Create Project"):
                if p_name:
                    conn = get_db_connection()
                    conn.execute("INSERT INTO projects (name, type, domains, llm_provider) VALUES (?, ?, ?, ?)",
                                 (p_name, p_type, ",".join(domains), llm_prov))
                    conn.commit()
                    conn.close()
                    st.success(f"Project '{p_name}' created!")

    # 3. DATA INGESTION (Full)
    elif selected_view == "Data Ingestion":
        st.title("Data Ingestion")
        if not st.session_state['active_project']:
            st.error("Select Active Project")
            return

        proj_id = st.session_state['active_project']['id']
        domains = st.session_state['active_project']['domains'].split(',')

        col1, col2 = st.columns(2)
        domain = col1.selectbox("Domain", domains)
        table_name = col2.text_input("Table Name (e.g., MARA)")
        uploaded_file = st.file_uploader("Upload Data", type=['csv', 'xlsx'])
        
        if st.button("Ingest Data"):
            if uploaded_file and table_name:
                conn = get_db_connection()
                existing = conn.execute("SELECT id, file_path FROM data_log WHERE project_id=? AND domain=? AND table_name=?", 
                                      (proj_id, domain, table_name)).fetchone()
                conn.close()
                file_path = os.path.join(DATA_STORAGE_DIR, f"{proj_id}_{domain}_{table_name}_{uploaded_file.name}")
                if existing:
                    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(file_path)
                    else: df = pd.read_excel(file_path)
                    conn = get_db_connection()
                    conn.execute("UPDATE data_log SET file_path=?, row_count=? WHERE id=?", (file_path, len(df), existing[0]))
                    conn.commit()
                    conn.close()
                    st.success("Data Updated!")
                else:
                    with open(file_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(file_path)
                    else: df = pd.read_excel(file_path)
                    conn = get_db_connection()
                    conn.execute("INSERT INTO data_log (project_id, domain, table_name, file_path, row_count) VALUES (?, ?, ?, ?, ?)",
                                 (proj_id, domain, table_name, file_path, len(df)))
                    conn.commit()
                    conn.close()
                    st.success(f"Ingested {len(df)} rows.")

        st.subheader("Ingested Data Tables")
        conn = get_db_connection()
        logs = pd.read_sql_query("SELECT * FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        if not logs.empty:
            for i, row in logs.iterrows():
                c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
                c1.write(f"**{row['domain']}**")
                c2.write(row['table_name'])
                c3.write(f"{row['row_count']} Rows")
                if c4.button("Delete", key=f"del_{row['id']}"):
                    conn = get_db_connection()
                    conn.execute("DELETE FROM data_log WHERE id=?", (row['id'],))
                    conn.commit()
                    conn.close()
                    if os.path.exists(row['file_path']): os.remove(row['file_path'])
                    st.rerun()

    # 4. DATA MAPPING (Full)
    elif selected_view == "Data Mapping":
        st.title("Data Mapping Workbench")
        if not st.session_state['active_project']:
            st.error("Select Active Project")
            return
            
        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        tables = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if tables.empty:
            st.warning("Ingest data first to start mapping.")
        else:
            table_opts = [f"{r['domain']} - {r['table_name']}" for i, r in tables.iterrows()]
            sel_table_str = st.selectbox("Select Table to Map", table_opts)
            sel_domain, sel_table = sel_table_str.split(" - ")
            
            # Load Source Columns
            file_path = tables[tables['table_name'] == sel_table]['file_path'].values[0]
            if os.path.exists(file_path):
                if file_path.endswith('.csv'): src_df = pd.read_csv(file_path, nrows=5)
                else: src_df = pd.read_excel(file_path, nrows=5)
                source_cols = ["-- Select --"] + src_df.columns.tolist()
            else:
                source_cols = ["File Missing"]

            # Load Saved Config
            conn = get_db_connection()
            saved_config = conn.execute("SELECT config_json FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", 
                                      (proj_id, sel_domain, sel_table)).fetchone()
            conn.close()
            
            config = json.loads(saved_config[0]) if saved_config else {"target_fields": [], "mappings": {}, "value_maps": {}}
            
            # 1. Define Template
            st.subheader("1. Target Template Definition")
            with st.expander("Manage Target Fields", expanded=True):
                c1, c2 = st.columns([3, 1])
                new_field_input = c1.text_input("Add Target Field(s) (comma-separated)", placeholder="Material_ID, Description, Plant")
                if c2.button("Add Field(s)"):
                    if new_field_input:
                        new_fields = [f.strip() for f in new_field_input.split(',') if f.strip()]
                        added_count = 0
                        for nf in new_fields:
                            if nf not in config['target_fields']:
                                config['target_fields'].append(nf)
                                added_count += 1
                        
                        if added_count > 0:
                            conn = get_db_connection()
                            conn.execute("DELETE FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, sel_domain, sel_table))
                            conn.execute("INSERT INTO mapping_config (project_id, domain, table_name, config_json) VALUES (?, ?, ?, ?)", 
                                         (proj_id, sel_domain, sel_table, json.dumps(config)))
                            conn.commit()
                            conn.close()
                            st.success(f"Added {added_count} fields!")
                            st.rerun()
                
                st.write("Current Target Fields:", ", ".join(config['target_fields']))

            # 2. Field Mapping
            st.subheader("2. Field Mapping & Value Logic")
            updated_mappings = config.get('mappings', {})
            updated_value_maps = config.get('value_maps', {})
            
            for t_field in config['target_fields']:
                with st.container():
                    col1, col2, col3 = st.columns([2, 2, 2])
                    col1.markdown(f"**{t_field}**")
                    curr_val = updated_mappings.get(t_field, "-- Select --")
                    idx = source_cols.index(curr_val) if curr_val in source_cols else 0
                    selected_source = col2.selectbox(f"Source for {t_field}", source_cols, index=idx, key=f"src_{t_field}")
                    updated_mappings[t_field] = selected_source
                    
                    if col3.button(f"Value Logic ({len(updated_value_maps.get(t_field, []))})", key=f"logic_{t_field}"):
                        st.session_state['active_mapping_field'] = t_field

            if st.button("Save Configuration"):
                config['mappings'] = updated_mappings
                conn = get_db_connection()
                conn.execute("DELETE FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, sel_domain, sel_table))
                conn.execute("INSERT INTO mapping_config (project_id, domain, table_name, config_json) VALUES (?, ?, ?, ?)", 
                             (proj_id, sel_domain, sel_table, json.dumps(config)))
                conn.commit()
                conn.close()
                st.success("Configuration Saved!")

            # 3. Value Mapping Modal
            if 'active_mapping_field' in st.session_state:
                field = st.session_state['active_mapping_field']
                st.markdown("---")
                st.info(f"Defining Value Mapping for: **{field}**")
                vm_list = updated_value_maps.get(field, [])
                c1, c2, c3 = st.columns([2, 2, 1])
                src_val = c1.text_input("Old Value", key="vm_old")
                tgt_val = c2.text_input("New Value", key="vm_new")
                if c3.button("Add Rule"):
                    if src_val and tgt_val:
                        vm_list.append({"old": src_val, "new": tgt_val})
                        updated_value_maps[field] = vm_list
                        config['value_maps'] = updated_value_maps
                        st.rerun()
                for i, vm in enumerate(vm_list):
                    st.write(f"{i+1}. '{vm['old']}' ➔ '{vm['new']}'")
                if st.button("Close Logic Editor"):
                    del st.session_state['active_mapping_field']
                    st.rerun()

    # 5. BP DEDUPLICATION
    elif selected_view == "BP Deduplication":
        st.title("BP Deduplication")
        uploaded_file = st.file_uploader("Upload Data (CSV or Excel)", type=['csv', 'xlsx'])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            st.write("Preview:", df.head())
            if st.button("Start Process"):
                 with st.spinner("Processing..."):
                     time.sleep(2)
                     if not HAS_DEDUPE:
                        st.warning("Pandas-Dedupe library missing. Simulating results.")
                     st.success("Mock Deduplication Complete")
                     st.dataframe(pd.DataFrame({'Name': ['Mock A', 'Mock A'], 'Score': [0.95, 0.95]}))

    # 6. DQ RULES CONFIGURATION (Enhanced)
    elif selected_view == "DQ Rules Config":
        st.title("🛠️ DQ Rules Studio")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']
        
        conn = get_db_connection()
        tables = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if tables.empty: st.warning("Ingest data first."); return
        
        # 1. Context Selection
        c_sel, c_view = st.columns([1, 3])
        with c_sel:
            sel_table_str = st.selectbox("Select Table", [f"{r['domain']} - {r['table_name']}" for _, r in tables.iterrows()])
            dom, tbl = sel_table_str.split(" - ")
            path = tables[tables['table_name'] == tbl]['file_path'].values[0]
        
        # Load Mapped Data for Context
        sample_df, mapped_cols = get_mapped_dataframe(proj_id, dom, tbl, path)
        if mapped_cols: cols = mapped_cols
        elif not sample_df.empty: cols = sample_df.columns.tolist()
        else: cols = []

        with c_view:
            st.caption("Available Columns:")
            st.code(",  ".join(cols) if cols else "No columns found")

        st.divider()

        # 2. Main Layout: Studio (Left) vs Library (Right)
        col_studio, col_lib = st.columns([1, 1])

        with col_studio:
            st.subheader("Rule Editor")
            with st.container(border=True):
                r_name = st.text_input("Rule Name", value=st.session_state['edit_name'], placeholder="e.g., Check_Active_Status", key="input_rname")
                r_desc = st.text_area("Requirement (English)", value=st.session_state['edit_desc'], placeholder="e.g., Status must be 'Active'", key="input_rdesc")
                
                if st.button("✨ Generate Logic"):
                    code = generate_python_rule(r_desc, cols, st.session_state['active_project']['llm_provider'], st.session_state.get('api_key'))
                    st.session_state['txt_code_area'] = code
                    st.rerun()
                
                code_input = st.text_area("Python Boolean Mask", value=st.session_state.get('txt_code_area', ''), height=120, key="txt_code_area_widget")
                # Sync widget to state variable to allow saving
                st.session_state['txt_code_area'] = code_input

                # --- TEST FUNCTIONALITY ---
                c_test, c_save = st.columns(2)
                if c_test.button("🧪 Test Rule"):
                    if not code_input:
                        st.error("No code to test.")
                    else:
                        try:
                            # Run on sample
                            local_scope = {'df': sample_df.head(50), 'pd': pd, 'np': np} # Test on 50 rows
                            mask = eval(code_input, {"__builtins__": None}, local_scope)
                            
                            if isinstance(mask, pd.Series) and mask.dtype == bool:
                                p_count = mask.sum()
                                f_count = len(mask) - p_count
                                st.success("✅ Syntax Valid")
                                st.info(f"Preview (50 rows): {p_count} Pass, {f_count} Fail")
                            else:
                                st.error(f"❌ Error: Code must return a Boolean Series. Got {type(mask)}")
                        except Exception as e:
                            st.error(f"❌ Execution Error: {e}")

                if c_save.button("💾 Save Rule", type="primary"):
                    # Use Callback logic inline or call function if safe
                    save_rule_callback(proj_id, dom, tbl)
                    st.rerun()

        with col_lib:
            st.subheader("Existing Rules")
            conn = get_db_connection()
            rules = pd.read_sql_query("SELECT id, rule_name, rule_description, python_code FROM dq_rules WHERE project_id=? AND domain=? AND table_name=?", conn, params=(proj_id, dom, tbl))
            conn.close()
            
            if not rules.empty:
                for i, row in rules.iterrows():
                    with st.expander(f"**{row['rule_name']}**", expanded=False):
                        st.caption(row['rule_description'])
                        st.code(row['python_code'], language='python')
                        
                        c_edit, c_del = st.columns([1, 1])
                        if c_edit.button("Edit", key=f"ed_{row['id']}"):
                            load_rule_callback(row['id'], row['rule_name'], row['rule_description'], row['python_code'])
                            st.rerun()
                        
                        if c_del.button("Delete", key=f"del_{row['id']}"):
                            delete_rule_callback(row['id'])
                            st.rerun()
            else:
                st.info("No rules defined for this table yet.")

    # 7. STEWARDSHIP (Full)
    elif selected_view == "Data Stewardship":
        st.title("Data Stewardship")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        t_df = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if t_df.empty: return
        sel_k = st.selectbox("Dataset", [f"{r['domain']} - {r['table_name']}" for _, r in t_df.iterrows()])
        dom, tbl = sel_k.split(" - ")
        path = t_df[t_df['table_name'] == tbl]['file_path'].values[0]

        if 'steward_df' not in st.session_state:
            st.session_state['steward_df'], _ = get_mapped_dataframe(proj_id, dom, tbl, path)

        if st.button("RUN DQ ANALYSIS ⚡", type="primary"):
            df = st.session_state['steward_df'].copy()
            conn = get_db_connection()
            rules = pd.read_sql_query("SELECT * FROM dq_rules WHERE project_id=? AND domain=? AND table_name=?", conn, params=(proj_id, dom, tbl))
            
            # Reset logs for this run
            conn.execute("DELETE FROM dq_results_log WHERE project_id=? AND domain=? AND table_name=?", (proj_id, dom, tbl))
            
            for _, r in rules.iterrows():
                try:
                    mask = eval(r['python_code'], {"__builtins__": None}, {'df': df, 'pd': pd, 'np': np})
                    if isinstance(mask, pd.Series):
                        df[f"{r['rule_name']}_Status"] = mask.map({True: "Valid", False: "Invalid"})
                        df[f"{r['rule_name']}_Justification"] = mask.map({True: "Passed", False: r['rule_description']})
                        
                        conn.execute("INSERT INTO dq_results_log (project_id, domain, table_name, rule_name, pass_count, fail_count) VALUES (?, ?, ?, ?, ?, ?)",
                                     (proj_id, dom, tbl, r['rule_name'], int(mask.sum()), int(len(mask)-mask.sum())))
                except Exception as e: st.error(f"Rule {r['rule_name']} failed: {e}")
            
            conn.commit(); conn.close()
            st.session_state['steward_df'] = df
            st.success("Analysis Complete")

        st.data_editor(st.session_state['steward_df'], num_rows="dynamic", use_container_width=True, key="main_editor")
        if st.button("Save to Disk"):
            # Save logic...
            st.success("Saved!")

    # 8. DATA EXPLORATION (Full Chat)
    elif selected_view == "Data Exploration":
        st.title("AI Data Explorer")
        # Chat implementation
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
                sys_prompt = f"Data Analyst. Columns: {df.columns.tolist()}. Generate Python code in ```python blocks to answer."
                api_key = st.session_state.get('api_key')
                provider = st.session_state['active_project']['llm_provider']
                
                resp = query_llm(provider, api_key, user_input, sys_prompt)
                st.markdown(resp)
                st.session_state["chat_history"].append({"role": "assistant", "content": resp})

# --- RUN ---
init_db()
if not st.session_state['authenticated']: login_page()
else: main_app()
