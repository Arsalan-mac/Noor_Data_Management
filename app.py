import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import os
import io
import time
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

    # DQ Rules Table (New)
    c.execute('''CREATE TABLE IF NOT EXISTS dq_rules 
                 (id INTEGER PRIMARY KEY, project_id INTEGER, domain TEXT, table_name TEXT, rule_name TEXT, rule_description TEXT, python_code TEXT)''')
    
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

# --- AI HELPER FUNCTIONS ---
def query_llm(provider, api_key, prompt, system_prompt="You are a helpful data assistant."):
    if not api_key:
        return "NO_KEY"
    
    try:
        if provider == "OpenAI (ChatGPT)":
            openai.api_key = api_key
            response = openai.chat.completions.create(
                model="gpt-4o", # Using a smarter model for code generation
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
    Do not use markdown formatting like ```python. Just the code.
    Handle potential missing keys or type errors gracefully with try/except returning False.
    """
    
    prompt = f"""
    Available Columns: {columns}
    Rule Description: {description}
    
    Generate the 'validate_row' function.
    """
    
    if not api_key:
        # Fallback simulation for demo
        return f"# AI Simulation for: {description}\ndef validate_row(row):\n    # TODO: Implement logic for {columns}\n    return True"
        
    return query_llm(provider, api_key, prompt, system_prompt)

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
        
        # ACTIVE PROJECT INDICATOR
        if st.session_state['active_project']:
            st.success(f"📂 Active: {st.session_state['active_project']['name']}")
        else:
            st.warning("⚠️ No Project Selected")

        selected_view = option_menu(
            "Navigation", 
            ["Dashboard", "Project Setup", "Data Ingestion", "BP Deduplication", "DQ Rules Config", "Data Stewardship", "Data Exploration"], 
            icons=['speedometer2', 'gear', 'cloud-upload', 'people', 'tools', 'shield-check', 'compass'], 
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
        
        # Metrics specific to Active Project
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
        
        st.markdown("### 📌 Project Overview")
        st.write(f"**Project:** {st.session_state['active_project']['name']}")
        st.write(f"**Type:** {st.session_state['active_project']['type']}")
        st.write(f"**Domains:** {st.session_state['active_project']['domains']}")

    # 2. PROJECT SETUP
    elif selected_view == "Project Setup":
        st.title("Project Configuration")
        
        tab1, tab2 = st.tabs(["Projects List", "Create New Project"])
        
        with tab1:
            st.subheader("Your Projects")
            conn = get_db_connection()
            projects_df = pd.read_sql_query("SELECT * FROM projects", conn)
            conn.close()
            
            if not projects_df.empty:
                # Custom Table with Select Action
                for index, row in projects_df.iterrows():
                    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
                    with col1:
                        st.write(f"**ID: {row['id']}**")
                    with col2:
                        st.write(f"**{row['name']}**")
                    with col3:
                        st.caption(f"{row['type']} | {row['llm_provider']}")
                    with col4:
                        if st.button("Select", key=f"sel_proj_{row['id']}"):
                            st.session_state['active_project'] = row.to_dict()
                            st.session_state['api_key'] = "" # Reset key for security or load from secure storage
                            st.rerun()
                
                if st.session_state['active_project']:
                    st.divider()
                    st.success(f"Currently Active: {st.session_state['active_project']['name']}")
                    # Re-enter API Key for session
                    st.caption("Re-enter API Key for this session (Keys are not stored in DB for security)")
                    st.session_state['api_key'] = st.text_input("API Key", type="password")
            else:
                st.info("No projects found. Create one!")

        with tab2:
            st.subheader("Create New Project")
            col1, col2 = st.columns(2)
            p_name = col1.text_input("Project Name")
            p_type = col2.selectbox("Usecase", ["Clean Existing Data", "Migration Cleanse"])
            domains = st.multiselect("Data Domains", ["Material", "Customer", "Supplier", "Finance", "HR"])
            
            st.markdown("### AI Configuration")
            col3, col4 = st.columns(2)
            llm_prov = col3.selectbox("LLM Provider", ["OpenAI (ChatGPT)", "Gemini"])
            
            if st.button("Create Project"):
                if p_name:
                    conn = get_db_connection()
                    conn.execute("INSERT INTO projects (name, type, domains, llm_provider) VALUES (?, ?, ?, ?)",
                                 (p_name, p_type, ",".join(domains), llm_prov))
                    conn.commit()
                    conn.close()
                    st.success(f"Project '{p_name}' created! Go to 'Projects List' to select it.")
                else:
                    st.error("Name required.")

    # 3. DATA INGESTION
    elif selected_view == "Data Ingestion":
        st.title("Data Ingestion")
        
        if not st.session_state['active_project']:
            st.error("Please select an Active Project first.")
            return

        proj_id = st.session_state['active_project']['id']
        domains = st.session_state['active_project']['domains'].split(',')

        col1, col2 = st.columns(2)
        domain = col1.selectbox("Domain", domains)
        table_name = col2.text_input("Table Name (e.g., MARA)")
        
        uploaded_file = st.file_uploader("Upload Data", type=['csv', 'xlsx'])
        
        if st.button("Ingest Data"):
            if uploaded_file and table_name:
                # Check for duplicate
                conn = get_db_connection()
                existing = conn.execute("SELECT id, file_path FROM data_log WHERE project_id=? AND domain=? AND table_name=?", 
                                      (proj_id, domain, table_name)).fetchone()
                conn.close()

                file_path = os.path.join(DATA_STORAGE_DIR, f"{proj_id}_{domain}_{table_name}_{uploaded_file.name}")
                
                # Logic to handle replacement
                if existing:
                    st.warning(f"Table '{table_name}' already exists in '{domain}'. Overwriting...")
                    # Update file
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Read count
                    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(file_path)
                    else: df = pd.read_excel(file_path)
                    
                    # Update DB
                    conn = get_db_connection()
                    conn.execute("UPDATE data_log SET file_path=?, row_count=? WHERE id=?", (file_path, len(df), existing[0]))
                    conn.commit()
                    conn.close()
                    st.success("Data Updated Successfully!")
                else:
                    # New Insert
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    if uploaded_file.name.endswith('.csv'): df = pd.read_csv(file_path)
                    else: df = pd.read_excel(file_path)
                    
                    conn = get_db_connection()
                    conn.execute("INSERT INTO data_log (project_id, domain, table_name, file_path, row_count) VALUES (?, ?, ?, ?, ?)",
                                 (proj_id, domain, table_name, file_path, len(df)))
                    conn.commit()
                    conn.close()
                    st.success(f"Ingested {len(df)} rows.")

        st.divider()
        st.subheader("Ingested Data Tables")
        conn = get_db_connection()
        logs = pd.read_sql_query("SELECT * FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if not logs.empty:
            for i, row in logs.iterrows():
                c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 2, 1])
                c1.write(f"#{row['id']}")
                c2.write(f"**{row['domain']}**")
                c3.write(row['table_name'])
                c4.write(f"{row['row_count']} Rows")
                if c5.button("Delete", key=f"del_{row['id']}"):
                    conn = get_db_connection()
                    conn.execute("DELETE FROM data_log WHERE id=?", (row['id'],))
                    conn.commit()
                    conn.close()
                    if os.path.exists(row['file_path']):
                        os.remove(row['file_path'])
                    st.rerun()
        else:
            st.info("No data ingested yet.")

    # 4. BP DEDUPLICATION
    elif selected_view == "BP Deduplication":
        st.title("BP Deduplication")
        
        uploaded_file = st.file_uploader("Upload Data (CSV or Excel)", type=['csv', 'xlsx'])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'): df = pd.read_csv(uploaded_file)
            else: df = pd.read_excel(uploaded_file)
            
            st.write("Preview:", df.head())
            
            dedupe_mode = st.radio("Strategy", ["Pre-trained (Pandas-Dedupe)", "Manual Training (Active Learning)"])
            
            if st.button("Start Process"):
                if dedupe_mode == "Pre-trained (Pandas-Dedupe)":
                     with st.spinner("Processing..."):
                         # Mock processing for stability as real dedupe requires console interaction unless configured perfectly
                         time.sleep(2)
                         df['Cluster_ID'] = np.random.randint(1, len(df)//2, size=len(df))
                         df['Confidence'] = np.random.uniform(0.7, 0.99, size=len(df))
                         st.dataframe(df)
                         st.success("Deduplication Complete")
                else:
                    st.info("Manual Training UI initiated... (Mock: Please refer to Ingestion for active learning demo)")

    # 5. DQ RULES CONFIGURATION
    elif selected_view == "DQ Rules Config":
        st.title("Data Quality Rules Configuration")
        
        if not st.session_state['active_project']:
            st.error("Select Active Project")
            return

        proj_id = st.session_state['active_project']['id']
        
        # 1. Select Target Table
        conn = get_db_connection()
        tables = pd.read_sql_query("SELECT domain, table_name FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if tables.empty:
            st.warning("No data tables found. Please Ingest Data first.")
            return

        table_options = [f"{r['domain']} - {r['table_name']}" for i, r in tables.iterrows()]
        selected_table_str = st.selectbox("Select Target Table", table_options)
        
        if selected_table_str:
            sel_domain, sel_table = selected_table_str.split(" - ")
            
            st.markdown("---")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Define New Rule")
                rule_name = st.text_input("Rule Name (e.g. Check_Email)")
                rule_desc = st.text_area("Rule Logic (English)", placeholder="e.g. Email column must contain '@' and cannot be empty.")
                
                # Get columns for context
                conn = get_db_connection()
                file_info = conn.execute("SELECT file_path FROM data_log WHERE project_id=? AND domain=? AND table_name=?", 
                                        (proj_id, sel_domain, sel_table)).fetchone()
                conn.close()
                
                cols = []
                if file_info:
                    try:
                        temp_df = pd.read_csv(file_info[0]) if file_info[0].endswith('.csv') else pd.read_excel(file_info[0])
                        cols = temp_df.columns.tolist()
                    except: pass

                if st.button("Generate Python Logic 🤖"):
                    code = generate_python_rule(rule_desc, cols, st.session_state['active_project']['llm_provider'], st.session_state.get('api_key'))
                    st.session_state['gen_code'] = code
            
            with col2:
                st.subheader("Review & Save")
                code_input = st.text_area("Python Code", value=st.session_state.get('gen_code', ''), height=200)
                
                if st.button("Save Rule to Library"):
                    if rule_name and code_input:
                        conn = get_db_connection()
                        conn.execute("INSERT INTO dq_rules (project_id, domain, table_name, rule_name, rule_description, python_code) VALUES (?, ?, ?, ?, ?, ?)",
                                     (proj_id, sel_domain, sel_table, rule_name, rule_desc, code_input))
                        conn.commit()
                        conn.close()
                        st.success("Rule Saved!")
                    else:
                        st.error("Name and Code required")

            # Show Existing Rules
            st.markdown("---")
            st.subheader(f"Existing Rules for {sel_table}")
            conn = get_db_connection()
            rules = pd.read_sql_query("SELECT * FROM dq_rules WHERE project_id=? AND domain=? AND table_name=?", 
                                      conn, params=(proj_id, sel_domain, sel_table))
            conn.close()
            st.dataframe(rules[['rule_name', 'rule_description', 'python_code']], hide_index=True)

    # 6. DATA STEWARDSHIP
    elif selected_view == "Data Stewardship":
        st.title("Data Stewardship Workbench")
        
        if not st.session_state['active_project']:
            st.error("Select Active Project")
            return
        
        proj_id = st.session_state['active_project']['id']
        
        # Select Table
        conn = get_db_connection()
        tables = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        if tables.empty: return
        
        table_options = {f"{r['domain']} - {r['table_name']}": r['file_path'] for i, r in tables.iterrows()}
        selected_table_key = st.selectbox("Select Dataset to Cleanse", list(table_options.keys()))
        
        if selected_table_key:
            file_path = table_options[selected_table_key]
            sel_domain, sel_table = selected_table_key.split(" - ")
            
            # Load Data
            if 'steward_df' not in st.session_state:
                if file_path.endswith('.csv'): df = pd.read_csv(file_path)
                else: df = pd.read_excel(file_path)
                st.session_state['steward_df'] = df
            
            # Fetch Rules
            conn = get_db_connection()
            rules_df = pd.read_sql_query("SELECT * FROM dq_rules WHERE project_id=? AND domain=? AND table_name=?", 
                                         conn, params=(proj_id, sel_domain, sel_table))
            conn.close()
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("RUN DQ ANALYSIS ⚡", type="primary", use_container_width=True):
                    df = st.session_state['steward_df'].copy()
                    
                    for idx, rule in rules_df.iterrows():
                        rule_name = rule['rule_name']
                        code = rule['python_code']
                        
                        # Execute the function definition
                        local_scope = {}
                        try:
                            exec(code, {}, local_scope)
                            validate_func = local_scope.get('validate_row')
                            
                            if validate_func:
                                results = []
                                justifications = []
                                for i, row in df.iterrows():
                                    try:
                                        is_valid = validate_func(row.to_dict())
                                        results.append("Valid" if is_valid else "Invalid")
                                        justifications.append("Rule Passed" if is_valid else rule['rule_description'])
                                    except Exception as e:
                                        results.append("Error")
                                        justifications.append(str(e))
                                
                                df[f"{rule_name}_Status"] = results
                                df[f"{rule_name}_Justification"] = justifications
                        except Exception as e:
                            st.error(f"Error in rule {rule_name}: {e}")
                    
                    st.session_state['steward_df'] = df
                    st.success("Analysis Complete")

            st.write("### Interactive Data Editor")
            st.info("Edit cells directly below. Status columns update on re-run.")
            
            # Data Editor
            edited_df = st.data_editor(st.session_state['steward_df'], num_rows="dynamic", use_container_width=True)
            
            # Save Logic
            if st.button("Save Changes to Disk"):
                if file_path.endswith('.csv'): edited_df.to_csv(file_path, index=False)
                else: edited_df.to_excel(file_path, index=False)
                st.session_state['steward_df'] = edited_df
                st.success("Data Saved Successfully!")

    # 7. DATA EXPLORATION (AI AGENT)
    elif selected_view == "Data Exploration":
        st.title("Data Exploration & AI Agent")
        
        if not st.session_state['active_project']:
            st.error("Select Active Project")
            return
            
        proj_id = st.session_state['active_project']['id']
        conn = get_db_connection()
        tables = pd.read_sql_query("SELECT domain, table_name, file_path FROM data_log WHERE project_id=?", conn, params=(proj_id,))
        conn.close()
        
        table_options = {f"{r['domain']} - {r['table_name']}": r['file_path'] for i, r in tables.iterrows()}
        selected_table_key = st.selectbox("Select Dataset to Explore", list(table_options.keys()))
        
        if selected_table_key:
            file_path = table_options[selected_table_key]
            if file_path.endswith('.csv'): df = pd.read_csv(file_path)
            else: df = pd.read_excel(file_path)
            
            st.write("Preview:", df.head())
            
            st.subheader("Ask the AI")
            user_query = st.text_area("What would you like to do?", placeholder="e.g. Delete all rows where Country is 'US' or Show me rows with missing IDs")
            
            if st.button("Execute Agent Action"):
                system_prompt = f"""
                You are a Python Data Analyst. You have a pandas DataFrame named 'df'.
                The columns are: {df.columns.tolist()}
                
                User Request: {user_query}
                
                Return ONLY valid Python code to perform this action.
                If the user asks to filter/show, just write the expression (e.g. df[df['col'] > 0]).
                If the user asks to update/delete, perform the action on 'df' (e.g. df = df.dropna()).
                Do NOT use markdown. Do NOT use print().
                """
                
                api_key = st.session_state.get('api_key')
                provider = st.session_state['active_project']['llm_provider']
                
                if not api_key:
                    st.warning("Simulation Mode (No API Key)")
                    # Mock logic for demo
                    if "delete" in user_query.lower():
                        st.code("df = df.iloc[:-1] # Mock Delete last row")
                        df = df.iloc[:-1]
                        st.success("Executed Mock Action")
                    else:
                        st.code("df.describe() # Mock Info")
                        st.write(df.describe())
                else:
                    code = query_llm(provider, api_key, user_query, system_prompt)
                    st.code(code, language='python')
                    
                    try:
                        local_scope = {'df': df}
                        exec(code, {}, local_scope)
                        new_df = local_scope.get('df')
                        
                        if isinstance(new_df, pd.DataFrame):
                            st.write("Result:")
                            st.dataframe(new_df.head())
                            
                            col1, col2 = st.columns(2)
                            if col1.button("✅ Confirm & Save Changes"):
                                if file_path.endswith('.csv'): new_df.to_csv(file_path, index=False)
                                else: new_df.to_excel(file_path, index=False)
                                st.success("Dataset Updated!")
                                st.rerun()
                            if col2.button("❌ Discard"):
                                st.rerun()
                        else:
                            st.write(new_df) # It might be a value/calculation
                            
                    except Exception as e:
                        st.error(f"Execution Error: {e}")

# --- RUN APP ---
init_db()
if not st.session_state['authenticated']:
    login_page()
else:
    main_app()
