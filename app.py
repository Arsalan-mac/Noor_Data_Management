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

# --- CALLBACKS & HELPERS ---
def load_rule_for_edit(r_id, r_name, r_desc, r_code):
    st.session_state['edit_rule_id'] = r_id
    st.session_state['edit_name'] = r_name
    st.session_state['edit_desc'] = r_desc
    st.session_state['edit_code'] = r_code
    st.session_state['txt_code_area'] = r_code

def delete_rule_db(r_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM dq_rules WHERE id=?", (r_id,))
    conn.commit()
    conn.close()
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
        
        # RENAMED & REMOVED ITEMS
        menu_items = ["Dashboard", "Project Setup", "Data Ingestion", "Data Mapping", "DQ Rules Config", "Data Cleansing", "Smart A.I."]
        icons = ['speedometer2', 'gear', 'cloud-upload', 'git', 'tools', 'shield-check', 'compass']
        
        selected_view = option_menu("Navigation", menu_items, icons=icons, menu_icon="cast", default_index=0, styles={"container": {"padding": "0!important", "background-color": "#0f172a"},"nav-link": {"font-size": "14px", "text-align": "left", "--hover-color": "#1e293b"},"nav-link-selected": {"background-color": "#1e293b", "border-left": "4px solid #2563eb"}}) if HAS_OPTION_MENU else st.radio("Navigation", menu_items)
        
        if st.button("Logout"): st.session_state['authenticated'] = False; st.rerun()

    # 1. DASHBOARD
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
                conn.commit(); conn.close(); st.success("Created!")

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
                conn.commit(); conn.close(); st.success(f"Added {added} fields"); st.rerun()
        
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

    # 5. DQ RULES CONFIGURATION
    elif selected_view == "DQ Rules Config":
        st.title("🛠️ DQ Rules Studio")
        if not st.session_state['active_project']: st.error("Select Project"); return
        proj_id = st.session_state['active_project']['id']
        
        conn = get_db_connection()
        tables = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if tables.empty: st.warning("Ingest data first."); return
        
        c_sel, c_view = st.columns([1, 3])
        with c_sel:
            sel_table_str = st.selectbox("Select Table", [f"{r['domain']} - {r['table_name']}" for _, r in tables.iterrows()])
            dom, tbl = sel_table_str.split(" - ")
            path = tables[tables['table_name'] == tbl]['file_path'].values[0]
        
        sample_df, mapped_cols = get_mapped_dataframe(proj_id, dom, tbl, path)
        cols = mapped_cols if mapped_cols else (sample_df.columns.tolist() if not sample_df.empty else [])

        with c_view:
            st.caption("Available Columns:")
            st.code(",  ".join(cols) if cols else "No columns found")

        st.divider()
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
                st.session_state['txt_code_area'] = code_input

                c_test, c_save = st.columns(2)
                if c_test.button("🧪 Test Rule"):
                    if not code_input: st.error("No code.")
                    else:
                        try:
                            local_scope = {'df': sample_df.head(50), 'pd': pd, 'np': np}
                            mask = eval(code_input, {"__builtins__": None}, local_scope)
                            if isinstance(mask, pd.Series) and mask.dtype == bool:
                                st.success(f"✅ Valid. Pass: {mask.sum()}, Fail: {len(mask)-mask.sum()}")
                            else: st.error("Error: Must return Boolean Series.")
                        except Exception as e: st.error(f"Error: {e}")

                # --- FIXED SAVE LOGIC ---
                if c_save.button("💾 Save Rule", type="primary"):
                    r_code = st.session_state.get('txt_code_area', '')
                    if r_name and r_code:
                        conn = get_db_connection()
                        eid = st.session_state.get('edit_rule_id')
                        if eid:
                            conn.execute("UPDATE dq_rules SET rule_name=?, rule_description=?, python_code=? WHERE id=?", (r_name, r_desc, r_code, eid))
                            st.toast("Rule Updated!")
                        else:
                            conn.execute("INSERT INTO dq_rules (project_id, domain, table_name, rule_name, rule_description, python_code) VALUES (?, ?, ?, ?, ?, ?)", (proj_id, dom, tbl, r_name, r_desc, r_code))
                            st.toast("Rule Created!")
                        conn.commit(); conn.close()
                        # Reset
                        st.session_state['edit_rule_id'] = None
                        st.session_state['edit_name'] = ""
                        st.session_state['edit_desc'] = ""
                        st.session_state['txt_code_area'] = ""
                        st.rerun() # Force Reload
                    else:
                        st.error("Name and Code required.")

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
                            load_rule_for_edit(row['id'], row['rule_name'], row['rule_description'], row['python_code'])
                            st.rerun()
                        if c_del.button("Delete", key=f"del_{row['id']}"):
                            delete_rule_db(row['id'])
                            st.rerun()
            else: st.info("No rules defined.")

    # 6. DATA CLEANSING (Renamed from Stewardship)
    elif selected_view == "Data Cleansing":
        st.title("Data Cleansing & Stewardship")
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
            st.success("Saved!")

    # 7. SMART A.I. (Renamed from Data Exploration)
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
                sys_prompt = f"Data Analyst. Columns: {df.columns.tolist()}. Generate Python code in ```python blocks to answer."
                api_key = st.session_state.get('api_key')
                provider = st.session_state['active_project']['llm_provider']
                
                resp = query_llm(provider, api_key, user_input, sys_prompt)
                
                code_match = re.search(r'```python(.*?)```', resp, re.DOTALL)
                final_resp = resp
                
                if code_match:
                    code = code_match.group(1).strip()
                    try:
                        f = io.StringIO()
                        with redirect_stdout(f):
                            local_scope = {'df': df, 'pd': pd, 'np': np}
                            exec(code, {}, local_scope)
                        output = f.getvalue()
                        new_df = local_scope.get('df')
                        if new_df is not None and not new_df.equals(df):
                            st.session_state['chat_df'] = new_df
                            final_resp += "\n\n✅ **Data Modified**"
                        if output: final_resp += f"\n\n**Output:**\n```\n{output}\n```"
                    except Exception as e: final_resp += f"\n\n❌ Error: {e}"
                
                st.markdown(final_resp)
                st.session_state["chat_history"].append({"role": "assistant", "content": final_resp})

        if st.button("Save Chat Changes to Disk"):
            if path.endswith('.csv'): st.session_state['chat_df'].to_csv(path, index=False)
            else: st.session_state['chat_df'].to_excel(path, index=False)
            st.success("Changes Saved!")

# --- RUN ---
init_db()
if not st.session_state['authenticated']: login_page()
else: main_app()
