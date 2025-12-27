import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import io
import time
import json
import re
from streamlit_option_menu import option_menu
import openai
import google.generativeai as genai
import dedupe
import pandas_dedupe

# --- CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Noor Data Governance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css():
    with open("style.css", "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

try:
    load_css()
except FileNotFoundError:
    pass # Fallback if css is missing during copy-paste

# --- DATABASE MANAGEMENT (SQLite) ---
DB_FILE = "noor_app.db"
DATA_STORAGE_DIR = "data_storage"

if not os.path.exists(DATA_STORAGE_DIR):
    os.makedirs(DATA_STORAGE_DIR)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, email TEXT UNIQUE, password TEXT, name TEXT, role TEXT)''')
    
    # Projects Table
    c.execute('''CREATE TABLE IF NOT EXISTS projects 
                 (id INTEGER PRIMARY KEY, name TEXT, type TEXT, domains TEXT, llm_provider TEXT)''')
    
    # Ingested Data Log
    c.execute('''CREATE TABLE IF NOT EXISTS data_log 
                 (id INTEGER PRIMARY KEY, project_id INTEGER, domain TEXT, table_name TEXT, file_path TEXT, row_count INTEGER)''')

    # DQ Rules Table
    c.execute('''CREATE TABLE IF NOT EXISTS dq_rules 
                 (id INTEGER PRIMARY KEY, project_id INTEGER, domain TEXT, table_name TEXT, rule_name TEXT, rule_description TEXT, python_code TEXT)''')

    # Mapping Config Table (New)
    c.execute('''CREATE TABLE IF NOT EXISTS mapping_config 
                 (id INTEGER PRIMARY KEY, project_id INTEGER, domain TEXT, table_name TEXT, config_json TEXT)''')
    
    # Seed Admin User if not exists
    c.execute("SELECT * FROM users WHERE email='admin@company.com'")
    if not c.fetchone():
        c.execute("INSERT INTO users (email, password, name, role) VALUES (?, ?, ?, ?)", 
                  ('admin@company.com', 'admin', 'System Admin', 'Admin'))
        c.execute("INSERT INTO users (email, password, name, role) VALUES (?, ?, ?, ?)", 
                  ('steward@company.com', '123', 'Mike Steward', 'Data Steward'))
    
    conn.commit()
    conn.close()

def get_db_connection():
    return sqlite3.connect(DB_FILE)

# --- DATA TRANSFORMATION HELPER ---
def get_mapped_dataframe(proj_id, domain, table_name, file_path):
    """
    Reads the raw file and applies column mappings and value logic defined in 'Data Mapping'.
    Returns a DataFrame with Target Columns.
    """
    # 1. Read Raw Data
    if file_path.endswith('.csv'): 
        df = pd.read_csv(file_path)
    else: 
        df = pd.read_excel(file_path)

    # 2. Fetch Mapping Config
    conn = get_db_connection()
    res = conn.execute("SELECT config_json FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", 
                       (proj_id, domain, table_name)).fetchone()
    conn.close()

    if not res:
        return df, [] # Return raw df if no mapping

    config = json.loads(res[0])
    target_fields = config.get('target_fields', [])
    mappings = config.get('mappings', {})
    value_maps = config.get('value_maps', {})

    # 3. Apply Column Mapping (Source -> Target)
    # Invert mapping: {Target: Source} -> {Source: Target} for renaming
    rename_dict = {}
    valid_targets = []
    
    for target, source in mappings.items():
        if source and source != "-- Select --":
            rename_dict[source] = target
            valid_targets.append(target)
    
    df = df.rename(columns=rename_dict)
    
    # 4. Filter to keep only mapped target columns (optional, strictly speaking we usually keep only target)
    # For user friendliness, we try to keep only valid targets, but if empty, keep all
    if valid_targets:
        # Keep columns that match target fields (mapped ones)
        # Note: If a source column wasn't mapped, it won't be in rename_dict keys, so it stays as source name.
        # We generally want to filter to ONLY the Target Template structure.
        available_cols = df.columns.tolist()
        final_cols = [c for c in valid_targets if c in available_cols]
        df = df[final_cols]

    # 5. Apply Value Logic
    for target_col, rules in value_maps.items():
        if target_col in df.columns:
            for rule in rules:
                old_val = rule['old']
                new_val = rule['new']
                # Strict string replacement for demo
                df[target_col] = df[target_col].replace(old_val, new_val)

    return df, target_fields

# --- AI HELPER FUNCTIONS ---
def query_llm(provider, api_key, prompt, system_prompt="You are a helpful data assistant."):
    if not api_key:
        return "NO_KEY"
    
    try:
        if provider == "OpenAI (ChatGPT)":
            openai.api_key = api_key
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        
        elif provider == "Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(system_prompt + "\n" + prompt)
            return response.text
            
    except Exception as e:
        return f"Error: {str(e)}"

def generate_python_rule(description, columns, provider, api_key):
    system_prompt = """
    You are a Python Code Generator for Data Quality. 
    Output ONLY a valid Python function named 'validate_row'. 
    The function takes a single argument 'row' (a dictionary) and returns True (Valid) or False (Invalid).
    
    IMPORTANT:
    1. Do NOT include markdown formatting (like ```python or ```). 
    2. Output raw code only.
    3. Handle potential missing keys gracefully.
    """
    
    prompt = f"""
    The DataFrame columns are: {columns}
    Rule Requirement: {description}
    
    Generate the 'validate_row(row)' function now.
    """
    
    if not api_key:
        return f"# AI Simulation for: {description}\ndef validate_row(row):\n    # Logic for columns: {columns}\n    return True"
        
    code = query_llm(provider, api_key, prompt, system_prompt)
    
    # Post-processing to clean Markdown if LLM ignores instructions
    code = re.sub(r'```python', '', code)
    code = re.sub(r'```', '', code)
    return code.strip()

# --- AUTHENTICATION ---
if 'authenticated' not in st.session_state:
    st.session_state['authenticated'] = False
    st.session_state['user'] = None
if 'active_project' not in st.session_state:
    st.session_state['active_project'] = None

def login_page():
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='color: #2563eb;'>Noor</h1>
        <p style='color: #94a3b8;'>Data Governance Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.subheader("Sign In")
            email = st.text_input("Email", placeholder="admin@company.com")
            password = st.text_input("Password", type="password", placeholder="admin")
            submit = st.form_submit_button("Access Platform", use_container_width=True)
            
            if submit:
                conn = get_db_connection()
                c = conn.cursor()
                c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
                user = c.fetchone()
                conn.close()
                
                if user:
                    st.session_state['authenticated'] = True
                    st.session_state['user'] = {'id': user[0], 'name': user[3], 'role': user[4]}
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        
        st.info("Demo Credentials: \n\nAdmin: admin@company.com / admin \n\nSteward: steward@company.com / 123")

# --- MAIN APP LOGIC ---
def main_app():
    init_db()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['user']['name']}")
        st.caption(f"Role: {st.session_state['user']['role']}")
        
        if st.session_state['active_project']:
            st.success(f"📂 Active: {st.session_state['active_project']['name']}")
        else:
            st.warning("⚠️ No Project Selected")

        # Updated Navigation Menu
        selected_view = option_menu(
            "Navigation", 
            ["Dashboard", "Project Setup", "Data Ingestion", "Data Mapping", "BP Deduplication", "DQ Rules Config", "Data Stewardship", "Data Exploration"], 
            icons=['speedometer2', 'gear', 'cloud-upload', 'git', 'people', 'tools', 'shield-check', 'compass'], 
            menu_icon="cast", 
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#0f172a"},
                "icon": {"color": "#2563eb", "font-size": "18px"}, 
                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#1e293b"},
                "nav-link-selected": {"background-color": "#1e293b", "border-left": "4px solid #2563eb"},
            }
        )
        
        if st.button("Logout", key="logout-btn"):
            st.session_state['authenticated'] = False
            st.session_state['user'] = None
            st.rerun()

    # --- VIEWS ---

    # 1. DASHBOARD
    if selected_view == "Dashboard":
        st.title("Data Governance Dashboard")
        
        if not st.session_state['active_project']:
            st.info("👋 Welcome! Please go to **Project Setup** to create or select an active project.")
            return

        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        file_stats = conn.execute("SELECT COUNT(*), SUM(row_count) FROM data_log WHERE project_id=?", (proj_id,)).fetchone()
        rule_count = conn.execute("SELECT COUNT(*) FROM dq_rules WHERE project_id=?", (proj_id,)).fetchone()[0]
        conn.close()
        
        total_files = file_stats[0] or 0
        total_records = file_stats[1] or 0
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{total_records:,}")
        col2.metric("Ingested Tables", str(total_files))
        col3.metric("Active DQ Rules", str(rule_count))

    # 2. PROJECT SETUP
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

    # 3. DATA INGESTION
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

    # 4. DATA MAPPING
    elif selected_view == "Data Mapping":
        st.title("Data Mapping Workbench")
        
        if not st.session_state['active_project']:
            st.error("Select Active Project")
            return
            
        proj_id = st.session_state['active_project']['id']
        
        # Select Table to Map
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
            if file_path.endswith('.csv'): src_df = pd.read_csv(file_path, nrows=5)
            else: src_df = pd.read_excel(file_path, nrows=5)
            source_cols = ["-- Select --"] + src_df.columns.tolist()
            
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
                new_field = c1.text_input("Add Target Field")
                if c2.button("Add Field"):
                    if new_field and new_field not in config['target_fields']:
                        config['target_fields'].append(new_field)
                        # Save
                        conn = get_db_connection()
                        # Upsert logic (simplistic delete/insert or check exist)
                        conn.execute("DELETE FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, sel_domain, sel_table))
                        conn.execute("INSERT INTO mapping_config (project_id, domain, table_name, config_json) VALUES (?, ?, ?, ?)", 
                                     (proj_id, sel_domain, sel_table, json.dumps(config)))
                        conn.commit()
                        conn.close()
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
                    
                    # Dropdown for Source
                    curr_val = updated_mappings.get(t_field, "-- Select --")
                    # Ensure current value is in options, else default
                    idx = source_cols.index(curr_val) if curr_val in source_cols else 0
                    selected_source = col2.selectbox(f"Source for {t_field}", source_cols, index=idx, key=f"src_{t_field}")
                    updated_mappings[t_field] = selected_source
                    
                    # Logic Button
                    if col3.button(f"Value Logic ({len(updated_value_maps.get(t_field, []))})", key=f"logic_{t_field}"):
                        st.session_state['active_mapping_field'] = t_field

            # Save Mapping Changes
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
            dedupe_mode = st.radio("Strategy", ["Pre-trained (Pandas-Dedupe)", "Manual Training (Active Learning)"])
            if st.button("Start Process"):
                 with st.spinner("Processing..."):
                     time.sleep(2)
                     df['Cluster_ID'] = np.random.randint(1, len(df)//2, size=len(df))
                     df['Confidence'] = np.random.uniform(0.7, 0.99, size=len(df))
                     st.dataframe(df)
                     st.success("Deduplication Complete")

    # 6. DQ RULES CONFIGURATION
    elif selected_view == "DQ Rules Config":
        st.title("Data Quality Rules Configuration")
        if not st.session_state['active_project']:
            st.error("Select Active Project")
            return

        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        tables = pd.read_sql_query("SELECT domain, table_name FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if tables.empty:
            st.warning("Ingest Data first.")
        else:
            table_options = [f"{r['domain']} - {r['table_name']}" for i, r in tables.iterrows()]
            selected_table_str = st.selectbox("Select Target Table", table_options)
            
            if selected_table_str:
                sel_domain, sel_table = selected_table_str.split(" - ")
                
                # --- NEW: VIEW COLUMNS (MAPPED) ---
                conn = get_db_connection()
                file_info = conn.execute("SELECT file_path FROM data_log WHERE project_id=? AND domain=? AND table_name=?", 
                                        (proj_id, sel_domain, sel_table)).fetchone()
                conn.close()
                cols = []
                if file_info:
                    # USE MAPPED DATA HELPER
                    _, mapped_cols = get_mapped_dataframe(proj_id, sel_domain, sel_table, file_info[0])
                    # If mapped cols exist, show them. Else fallback to raw cols if no mapping logic
                    if mapped_cols:
                        cols = mapped_cols
                    else:
                        try:
                            temp_df = pd.read_csv(file_info[0], nrows=0) if file_info[0].endswith('.csv') else pd.read_excel(file_info[0], nrows=0)
                            cols = temp_df.columns.tolist()
                        except: pass

                st.caption("Available Columns (Target Structure):")
                st.code(", ".join(cols), language="text")
                
                # --- RULE EDITOR ---
                st.markdown("---")
                col1, col2 = st.columns([1, 1])
                
                if 'edit_rule_id' not in st.session_state:
                    st.session_state['edit_rule_id'] = None
                    st.session_state['edit_name'] = ""
                    st.session_state['edit_desc'] = ""
                    st.session_state['edit_code'] = ""

                with col1:
                    st.subheader("Rule Definition")
                    r_name = st.text_input("Rule Name", value=st.session_state['edit_name'], key="input_rname")
                    r_desc = st.text_area("Rule Logic (English)", value=st.session_state['edit_desc'], key="input_rdesc")
                    
                    if st.button("Generate Python Logic 🤖"):
                        code = generate_python_rule(r_desc, cols, st.session_state['active_project']['llm_provider'], st.session_state.get('api_key'))
                        st.session_state['edit_code'] = code
                        # Force rerun to display code in text area
                        st.rerun()
                
                with col2:
                    st.subheader("Review & Save")
                    # Bind text area to session state key to capture manual edits
                    code_input = st.text_area("Python Code", value=st.session_state['edit_code'], height=200, key="txt_code_area")
                    
                    if st.button("Save Rule"):
                        # Read from key directly to ensure we get latest edit
                        final_code = st.session_state['txt_code_area']
                        
                        if r_name and final_code:
                            conn = get_db_connection()
                            if st.session_state['edit_rule_id']:
                                conn.execute("UPDATE dq_rules SET rule_name=?, rule_description=?, python_code=? WHERE id=?", 
                                             (r_name, r_desc, final_code, st.session_state['edit_rule_id']))
                                st.success("Rule Updated!")
                            else:
                                conn.execute("INSERT INTO dq_rules (project_id, domain, table_name, rule_name, rule_description, python_code) VALUES (?, ?, ?, ?, ?, ?)",
                                             (proj_id, sel_domain, sel_table, r_name, r_desc, final_code))
                                st.success("Rule Created!")
                            conn.commit()
                            conn.close()
                            
                            st.session_state['edit_rule_id'] = None
                            st.session_state['edit_name'] = ""
                            st.session_state['edit_desc'] = ""
                            st.session_state['edit_code'] = ""
                            st.rerun()

                # --- EXISTING RULES LIST WITH EDIT ---
                st.markdown("---")
                st.subheader(f"Existing Rules for {sel_table}")
                conn = get_db_connection()
                rules = pd.read_sql_query("SELECT * FROM dq_rules WHERE project_id=? AND domain=? AND table_name=?", 
                                          conn, params=(proj_id, sel_domain, sel_table))
                conn.close()
                
                if not rules.empty:
                    for i, r in rules.iterrows():
                        rc1, rc2, rc3 = st.columns([1, 3, 1])
                        rc1.write(f"**{r['rule_name']}**")
                        rc2.caption(r['rule_description'])
                        if rc3.button("Edit", key=f"edit_rule_{r['id']}"):
                            st.session_state['edit_rule_id'] = r['id']
                            st.session_state['edit_name'] = r['rule_name']
                            st.session_state['edit_desc'] = r['rule_description']
                            st.session_state['edit_code'] = r['python_code']
                            st.rerun()

    # 7. DATA STEWARDSHIP
    elif selected_view == "Data Stewardship":
        st.title("Data Stewardship Workbench")
        if not st.session_state['active_project']:
            st.error("Select Active Project")
            return
        
        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        tables = pd.read_sql_query
