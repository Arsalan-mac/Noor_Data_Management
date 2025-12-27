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

# --- SAFE IMPORTS (Prevents Blank Screen if libs are missing) ---
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

# --- CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="Noor Data Governance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
def load_css():
    st.markdown("""
        <style>
        .stApp { background-color: #0f172a; color: #cbd5e1; }
        .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stTextArea textarea {
            background-color: #1e293b !important; color: #e2e8f0 !important; border: 1px solid #334155 !important;
        }
        div[data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }
        div[data-testid="stMetricValue"] { color: #f8fafc; }
        </style>
    """, unsafe_allow_html=True)
    
    if os.path.exists("style.css"):
        with open("style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

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

    # Mapping Config Table
    c.execute('''CREATE TABLE IF NOT EXISTS mapping_config 
                 (id INTEGER PRIMARY KEY, project_id INTEGER, domain TEXT, table_name TEXT, config_json TEXT)''')
    
    # DQ Results Log (New for Dashboard)
    c.execute('''CREATE TABLE IF NOT EXISTS dq_results_log 
                 (id INTEGER PRIMARY KEY, project_id INTEGER, domain TEXT, table_name TEXT, rule_name TEXT, pass_count INTEGER, fail_count INTEGER, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    
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
    if not os.path.exists(file_path):
        return pd.DataFrame(), []

    if file_path.endswith('.csv'): df = pd.read_csv(file_path)
    else: df = pd.read_excel(file_path)

    conn = get_db_connection()
    res = conn.execute("SELECT config_json FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", 
                       (proj_id, domain, table_name)).fetchone()
    conn.close()

    if not res: return df, [] 

    config = json.loads(res[0])
    target_fields = config.get('target_fields', [])
    mappings = config.get('mappings', {})
    value_maps = config.get('value_maps', {})

    rename_dict = {}
    valid_targets = []
    for target, source in mappings.items():
        if source and source != "-- Select --":
            rename_dict[source] = target
            valid_targets.append(target)
    
    df = df.rename(columns=rename_dict)
    
    if valid_targets:
        available_cols = df.columns.tolist()
        final_cols = [c for c in valid_targets if c in available_cols]
        df = df[final_cols]

    for target_col, rules in value_maps.items():
        if target_col in df.columns:
            for rule in rules:
                df[target_col] = df[target_col].replace(rule['old'], rule['new'])

    return df, target_fields

# --- AI HELPER FUNCTIONS ---
def query_llm(provider, api_key, prompt, system_prompt="You are a helpful data assistant."):
    if not api_key: return "NO_KEY"
    try:
        if provider == "OpenAI (ChatGPT)":
            if not HAS_OPENAI: return "Error: OpenAI library not installed."
            openai.api_key = api_key
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        elif provider == "Gemini":
            if not HAS_GEMINI: return "Error: Google GenAI library not installed."
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(system_prompt + "\n" + prompt)
            return response.text
    except Exception as e:
        return f"Error: {str(e)}"

def generate_python_rule(description, columns, provider, api_key):
    system_prompt = (
        "You are a data quality assistant. Convert the requirement to a Pandas boolean mask expression. "
        "Do NOT wrap it in df[...]. Return ONLY the boolean expression."
    )
    prompt = f"Columns: {columns}\nRequirement: {description}\nExpression:"
    
    if not api_key:
        col = columns[0] if columns else 'Col'
        return f"# Simulation\n(df['{col}'].notna())"
        
    code = query_llm(provider, api_key, prompt, system_prompt)
    code = re.sub(r'```python', '', code).replace('```', '').strip()
    return code

# --- CALLBACKS ---
def save_rule_callback(proj_id, sel_domain, sel_table):
    r_name = st.session_state.get('input_rname', '')
    r_desc = st.session_state.get('input_rdesc', '')
    final_code = st.session_state.get('txt_code_area', '')
    if r_name and final_code:
        conn = get_db_connection()
        if st.session_state['edit_rule_id']:
            conn.execute("UPDATE dq_rules SET rule_name=?, rule_description=?, python_code=? WHERE id=?", 
                            (r_name, r_desc, final_code, st.session_state['edit_rule_id']))
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
        st.info("Demo: admin@company.com / admin | steward@company.com / 123")

# --- MAIN APP ---
def main_app():
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state['user']['name']}")
        st.caption(f"Role: {st.session_state['user']['role']}")
        if st.session_state['active_project']: st.success(f"📂 Active: {st.session_state['active_project']['name']}")
        else: st.warning("⚠️ No Project Selected")
        
        menu_items = ["Dashboard", "Project Setup", "Data Ingestion", "Data Mapping", "BP Deduplication", "DQ Rules Config", "Data Stewardship", "Data Exploration"]
        icons = ['speedometer2', 'gear', 'cloud-upload', 'git', 'people', 'tools', 'shield-check', 'compass']
        selected_view = option_menu("Navigation", menu_items, icons=icons, menu_icon="cast", default_index=0, styles={"container": {"padding": "0!important", "background-color": "#0f172a"},"icon": {"color": "#2563eb", "font-size": "18px"},"nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "--hover-color": "#1e293b"},"nav-link-selected": {"background-color": "#1e293b", "border-left": "4px solid #2563eb"}}) if HAS_OPTION_MENU else st.radio("Navigation", menu_items)
        if st.button("Logout"): st.session_state['authenticated'] = False; st.rerun()

    # 1. DASHBOARD
    if selected_view == "Dashboard":
        st.title("Data Governance Dashboard")
        if not st.session_state['active_project']: st.info("Please select an Active Project."); return

        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        
        # Metrics Calculation
        tables_df = pd.read_sql_query("SELECT domain, table_name, row_count FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        rule_count = conn.execute("SELECT COUNT(*) FROM dq_rules WHERE project_id=?", (proj_id,)).fetchone()[0]
        results_df = pd.read_sql_query("SELECT rule_name, pass_count, fail_count FROM dq_results_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()

        # Overview Stats
        domains = tables_df['domain'].unique().tolist()
        total_rows = tables_df['row_count'].sum() if not tables_df.empty else 0
        total_tables = len(tables_df)
        
        # DQ Score Logic
        dq_score = 100
        if not results_df.empty:
            total_checks = results_df['pass_count'].sum() + results_df['fail_count'].sum()
            if total_checks > 0:
                dq_score = int((results_df['pass_count'].sum() / total_checks) * 100)

        # -- DISPLAY DOMAINS (New Visual) --
        if domains:
            st.info(f"**Applicable Domains:** {', '.join(domains)}")
        else:
            st.warning("No Domains Applicable yet (No data ingested).")

        # Top Metric Cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ingested Tables", total_tables)
        c2.metric("Total Rows", f"{total_rows:,}")
        c3.metric("Active Rules", rule_count)
        c4.metric("Overall DQ Score", f"{dq_score}%", delta_color="normal")

        col1, col2 = st.columns([1, 1])
        
        # Rows per Table
        with col1:
            st.subheader("Rows per Table")
            if not tables_df.empty:
                st.dataframe(tables_df[['domain', 'table_name', 'row_count']], use_container_width=True, hide_index=True)
            else:
                st.info("No data ingested.")

        # DQ Breakdown
        with col2:
            st.subheader("DQ Breakdown (Failures per Rule)")
            if not results_df.empty:
                # Group by rule name to aggregate latest runs
                rule_stats = results_df.groupby('rule_name')[['fail_count']].sum().reset_index()
                st.bar_chart(rule_stats.set_index('rule_name'))
            else:
                st.info("No DQ Analysis run yet.")

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
                conn.commit(); conn.close(); st.success("Created!")

    # 3. DATA INGESTION
    elif selected_view == "Data Ingestion":
        st.title("Data Ingestion")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']
        c1, c2 = st.columns(2)
        domain = c1.selectbox("Domain", st.session_state['active_project']['domains'].split(','))
        table = c2.text_input("Table Name")
        up_file = st.file_uploader("File", type=['csv', 'xlsx'])
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
            conn.commit(); conn.close(); st.success("Ingested!")
        
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
        st.title("Data Mapping")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        tables = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        if tables.empty: st.warning("No Data Ingested"); return
        
        t_opt = [f"{r['domain']} - {r['table_name']}" for _, r in tables.iterrows()]
        sel = st.selectbox("Select Table", t_opt)
        dom, tbl = sel.split(" - ")
        path = tables[tables['table_name'] == tbl]['file_path'].values[0]
        
        try: src_df = pd.read_csv(path, nrows=5) if path.endswith('.csv') else pd.read_excel(path, nrows=5)
        except: src_df = pd.DataFrame()
        src_cols = ["-- Select --"] + src_df.columns.tolist()

        conn = get_db_connection()
        saved = conn.execute("SELECT config_json FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, dom, tbl)).fetchone()
        conn.close()
        config = json.loads(saved[0]) if saved else {"target_fields": [], "mappings": {}, "value_maps": {}}

        st.subheader("1. Target Fields")
        c1, c2 = st.columns([3, 1])
        new_f = c1.text_input("Add Fields (comma-separated)", placeholder="ID, Name, Status")
        if c2.button("Add"):
            added = 0
            for f in new_f.split(','):
                f = f.strip()
                if f and f not in config['target_fields']: config['target_fields'].append(f); added+=1
            if added:
                conn = get_db_connection()
                conn.execute("DELETE FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, dom, tbl))
                conn.execute("INSERT INTO mapping_config (project_id, domain, table_name, config_json) VALUES (?, ?, ?, ?)", (proj_id, dom, tbl, json.dumps(config)))
                conn.commit(); conn.close(); st.rerun()
        
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
            conn.execute("DELETE FROM mapping_config WHERE project_id=? AND domain=? AND table_name=?", (proj_id, dom, tbl))
            conn.execute("INSERT INTO mapping_config (project_id, domain, table_name, config_json) VALUES (?, ?, ?, ?)", (proj_id, dom, tbl, json.dumps(config)))
            conn.commit(); conn.close(); st.success("Saved!")

        if 'active_mapping_field' in st.session_state:
            f = st.session_state['active_mapping_field']
            st.info(f"Value Mapping: {f}")
            c1, c2, c3 = st.columns([2, 2, 1])
            o = c1.text_input("Old"); n = c2.text_input("New")
            if c3.button("Add Rule") and o and n:
                l = config['value_maps'].get(f, []); l.append({"old": o, "new": n}); config['value_maps'][f] = l; st.rerun()
            for vm in config['value_maps'].get(f, []): st.write(f"{vm['old']} -> {vm['new']}")
            if st.button("Close"): del st.session_state['active_mapping_field']; st.rerun()

    # 5. BP DEDUPLICATION (Mock)
    elif selected_view == "BP Deduplication":
        st.title("BP Deduplication")
        f = st.file_uploader("Upload", type=['csv', 'xlsx'])
        if f and st.button("Start"):
            st.success("Mock Deduplication Complete")
            st.dataframe(pd.DataFrame({'Name': ['A', 'A'], 'Score': [0.95, 0.95]}))

    # 6. DQ RULES
    elif selected_view == "DQ Rules Config":
        st.title("DQ Rules")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        t_df = pd.read_sql_query("SELECT domain, table_name FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if t_df.empty: st.warning("Ingest data first."); return
        sel = st.selectbox("Table", [f"{r['domain']} - {r['table_name']}" for _, r in t_df.iterrows()])
        dom, tbl = sel.split(" - ")
        
        # Cols
        conn = get_db_connection()
        path = conn.execute("SELECT file_path FROM data_log WHERE project_id=? AND domain=? AND table_name=?", (proj_id, dom, tbl)).fetchone()[0]
        conn.close()
        _, cols = get_mapped_dataframe(proj_id, dom, tbl, path)
        if not cols:
             try: cols = pd.read_csv(path, nrows=0).columns.tolist() if path.endswith('.csv') else pd.read_excel(path, nrows=0).columns.tolist()
             except: cols = []
        st.code(", ".join(cols) if cols else "No columns found")

        c1, c2 = st.columns(2)
        rn = c1.text_input("Name", value=st.session_state.get('edit_name', ''), key="input_rname")
        rd = c1.text_area("Logic", value=st.session_state.get('edit_desc', ''), key="input_rdesc")
        if c1.button("Generate Python"):
            code = generate_python_rule(rd, cols, st.session_state['active_project']['llm_provider'], st.session_state.get('api_key'))
            st.session_state['txt_code_area'] = code
            st.rerun()
        
        c2.text_area("Code", height=200, key="txt_code_area")
        c2.button("Save", on_click=save_rule_callback, args=(proj_id, dom, tbl))

        conn = get_db_connection()
        rules = pd.read_sql_query("SELECT * FROM dq_rules WHERE project_id=? AND domain=? AND table_name=?", conn, params=(proj_id, dom, tbl))
        conn.close()
        for _, r in rules.iterrows():
            c1, c2, c3, c4 = st.columns([1, 3, 0.5, 0.5])
            c1.write(f"**{r['rule_name']}**"); c2.caption(r['rule_description'])
            c3.button("Edit", key=f"e_{r['id']}", on_click=load_rule_callback, args=(r['id'], r['rule_name'], r['rule_description'], r['python_code']))
            c4.button("❌", key=f"d_{r['id']}", on_click=delete_rule_callback, args=(r['id'],))

    # 7. DATA STEWARDSHIP
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

        if st.button("RUN DQ ANALYSIS ⚡"):
            df = st.session_state['steward_df'].copy()
            conn = get_db_connection()
            rules = pd.read_sql_query("SELECT * FROM dq_rules WHERE project_id=? AND domain=? AND table_name=?", conn, params=(proj_id, dom, tbl))
            
            # Clear old logs for this table to avoid double counting in dashboard (optional strategy)
            # conn.execute("DELETE FROM dq_results_log WHERE project_id=? AND domain=? AND table_name=?", (proj_id, dom, tbl))
            
            for _, r in rules.iterrows():
                try:
                    mask = eval(r['python_code'], {"__builtins__": None}, {'df': df, 'pd': pd, 'np': np})
                    if isinstance(mask, pd.Series):
                        df[f"{r['rule_name']}_Status"] = mask.map({True: "Valid", False: "Invalid"})
                        df[f"{r['rule_name']}_Justification"] = mask.map({True: "Rule Passed", False: r['rule_description']})
                        
                        # Log Results for Dashboard
                        pass_c = mask.sum()
                        fail_c = len(mask) - pass_c
                        conn.execute("INSERT INTO dq_results_log (project_id, domain, table_name, rule_name, pass_count, fail_count) VALUES (?, ?, ?, ?, ?, ?)",
                                     (proj_id, dom, tbl, r['rule_name'], int(pass_c), int(fail_c)))
                except: pass
            
            conn.commit()
            conn.close()
            st.session_state['steward_df'] = df
            st.success("Analysis Complete & Logged")

        edited = st.data_editor(st.session_state['steward_df'], num_rows="dynamic", use_container_width=True)
        if st.button("Save to Disk"):
            if path.endswith('.csv'): edited.to_csv(path, index=False)
            else: edited.to_excel(path, index=False)
            st.session_state['steward_df'] = edited
            st.success("Saved!")

    # 8. DATA EXPLORATION (CHATBOT)
    elif selected_view == "Data Exploration":
        st.title("Data Exploration & AI Agent")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']
        
        # 1. Select Context
        conn = get_db_connection()
        t_df = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if t_df.empty: st.warning("No data found."); return
        sel_k = st.selectbox("Select Context (Table)", [f"{r['domain']} - {r['table_name']}" for _, r in t_df.iterrows()])
        dom, tbl = sel_k.split(" - ")
        path = t_df[t_df['table_name'] == tbl]['file_path'].values[0]

        # Load Data
        if 'chat_df' not in st.session_state:
            st.session_state['chat_df'], _ = get_mapped_dataframe(proj_id, dom, tbl, path)
        
        # Chat History
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = [{"role": "assistant", "content": "Hello! I have loaded your data. Ask me anything or tell me to modify it."}]

        # Display Chat
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        if user_input := st.chat_input("Ask about data or request changes..."):
            st.session_state["chat_history"].append({"role": "user", "content": user_input})
            with st.chat_message("user"): st.markdown(user_input)

            # AI Logic
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                # Construct Prompt
                df = st.session_state['chat_df']
                cols = df.columns.tolist()
                head_csv = df.head(3).to_csv(index=False)
                
                sys_prompt = f"""
                You are a Data Analyst Agent. You have a pandas dataframe 'df'.
                Columns: {cols}
                Preview:
                {head_csv}
                
                INSTRUCTIONS:
                1. Answer questions naturally.
                2. Only generate Python code (wrapped in ```python ... ```) if data manipulation or complex calculation is explicitly required.
                3. If modifying data, apply changes to 'df'.
                4. ALWAYS print the result of your calculation using `print()`.
                """
                
                api_key = st.session_state.get('api_key')
                provider = st.session_state['active_project']['llm_provider']
                
                response_text = query_llm(provider, api_key, user_input, sys_prompt)
                
                # Check for code
                code_match = re.search(r'```python(.*?)```', response_text, re.DOTALL)
                
                final_response = response_text
                
                if code_match:
                    code = code_match.group(1).strip()
                    # Execute Code
                    try:
                        # Capture stdout
                        f = io.StringIO()
                        with redirect_stdout(f):
                            local_scope = {'df': df, 'pd': pd, 'np': np}
                            exec(code, {}, local_scope)
                        
                        output = f.getvalue()
                        
                        # Check if DF changed
                        new_df = local_scope.get('df')
                        if new_df is not None and not new_df.equals(df):
                            st.session_state['chat_df'] = new_df
                            final_response += "\n\n✅ **Data Modified Successfully.**"
                            
                        if output:
                            final_response += f"\n\n**Output:**\n```\n{output}\n```"
                            
                    except Exception as e:
                        final_response += f"\n\n❌ **Execution Error:** {str(e)}"
                
                message_placeholder.markdown(final_response)
                st.session_state["chat_history"].append({"role": "assistant", "content": final_response})

        # Save Button for Chat Changes
        if st.button("Save Chat Changes to Disk"):
            if path.endswith('.csv'): st.session_state['chat_df'].to_csv(path, index=False)
            else: st.session_state['chat_df'].to_excel(path, index=False)
            st.success("Changes Saved!")

# --- RUN ---
init_db()
if not st.session_state['authenticated']: login_page()
else: main_app()
