import streamlit as st
import graphviz
import json
import os
import random
from dotenv import load_dotenv

# 1. SETUP & CONFIG
load_dotenv()

# Detect Mode (Env Var -> Secrets Fallback)
PROJECT_ID = os.getenv("PROJECT_ID") or st.secrets.get("PROJECT_ID")
DD_API_KEY = os.getenv("DD_API_KEY") or st.secrets.get("DD_API_KEY")
DD_SITE = os.getenv("DD_SITE") or st.secrets.get("DD_SITE")
MODEL = os.getenv("MODEL") or st.secrets.get("MODEL")

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
import numpy as np
from ddtrace import tracer, patch_all
from ddtrace.llmobs import LLMObs

# Optional interactive graph
try:
    from streamlit_agraph import agraph, Node, Edge, Config
except ImportError:
    agraph = None
    Node = None
    Edge = None
    Config = None

patch_all()

# Initialize Datadog
if DD_API_KEY:
    LLMObs.enable(
        ml_app=os.getenv("DD_SERVICE"),
        api_key=DD_API_KEY,
        site=os.getenv("DD_SITE")
    )

# Initialize Google Vertex AI
# AUTHENTICATION FIX: Support Streamlit Secrets (Service Account)
from google.oauth2 import service_account

credentials = None
if "gcp_service_account" in st.secrets:
    print("DEBUG: Found [gcp_service_account] in secrets")
    # Create credentials from the secrets dictionary
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
elif "type" in st.secrets and st.secrets["type"] == "service_account":
    print("DEBUG: Found flattened service account in secrets")
    # Handle case where secrets are flattened (no [gcp_service_account] section)
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
else:
    print("DEBUG: No service account found in secrets")

if credentials:
    print("DEBUG: Initializing vertexai with credentials")
    vertexai.init(project=PROJECT_ID, location="us-central1", credentials=credentials)
    # Also initialize aiplatform explicitly to ensure Model Garden uses the credentials
    from google.cloud import aiplatform
    aiplatform.init(project=PROJECT_ID, location="us-central1", credentials=credentials)
else:
    # Fallback to default (CLI/Environment) auth
    print("DEBUG: Initializing vertexai without credentials (ADC)")
    vertexai.init(project=PROJECT_ID, location="us-central1")

model = GenerativeModel(MODEL)

# Load the lightweight embedding model
print("DEBUG: Loading embedding model...")
embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")
print("DEBUG: Embedding model loaded successfully")

def count_local_tokens(text):
    """
    Fast, free local estimation (1 token ~= 4 chars).
    """
    if not text: return 0
    return len(text) // 4

def get_batch_embeddings(texts, task_type="RETRIEVAL_DOCUMENT"):
    """
    Generates vectors for a LIST of texts in ONE API call.
    This avoids hitting the '5 Requests Per Minute' limit.
    """
    if not texts: return []

    # Clean empty strings to avoid API errors
    valid_texts = [t if t else " " for t in texts]

    try:
        # Batch Call: Sends all texts at once!
        # Note: auto_truncate is True by default which handles long texts
        embeddings = embedding_model.get_embeddings(valid_texts)
        return [e.values for e in embeddings]
    except Exception as e:
        print(f"Embedding Error: {e}")
        # Fallback to empty vectors if batch fails
        return [np.zeros(3072) for _ in valid_texts]

def cosine_similarity(a, b):
    """Calculates semantic similarity (0 to 1)."""
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Wrapper to handle LLM tasks conditionally
def llm_task(name):
    def decorator(func):
        from ddtrace.llmobs import LLMObs
        def wrapper(*args, **kwargs):
            with LLMObs.task(name=name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# --- OBSERVABILITY HELPERS ---

def log_execution_metrics(response):
    """
    Extracts actual usage and calculates 'Shadow Savings' (what we avoided sending).
    """
    try:
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        total_tokens = input_tokens + output_tokens

        # Cost Calculation (Gemini Flash Pricing)
        input_cost = (input_tokens / 1000) * 0.00001875
        output_cost = (output_tokens / 1000) * 0.000075
        total_cost = input_cost + output_cost

        # Update Totals
        if "total_tokens" in st.session_state: st.session_state.total_tokens += total_tokens
        if "total_cost" in st.session_state: st.session_state.total_cost += total_cost

        # --- REAL-TIME SAVINGS CALCULATION ---
        # 1. Identify the Active Path (Nodes we JUST sent)
        active_path = set()
        curr = st.session_state.current_branch
        while curr:
            active_path.add(curr)
            curr = st.session_state.nodes[curr].get("parent")

        # 2. Sum tokens of INACTIVE branches (The "Pruned" Context)
        inactive_tokens = 0
        for name, node in st.session_state.nodes.items():
            if name not in active_path and name != "ROOT":
                # Count tokens in this inactive branch's history
                history_text = " ".join([m["content"] for m in node.get("history", [])])
                inactive_tokens += count_local_tokens(history_text)

        # 3. Update the Metric
        if "tokens_saved" not in st.session_state: st.session_state.tokens_saved = 0
        st.session_state.tokens_saved += inactive_tokens

        # Datadog
        from ddtrace import tracer
        span = tracer.current_span()
        if span:
            span.set_metric("arbor.tokens.total", total_tokens)
            span.set_metric("arbor.tokens.saved_real", inactive_tokens) # Real metric!
            span.set_metric("arbor.tokens.input", input_tokens)
            span.set_metric("arbor.tokens.output", output_tokens)
            span.set_metric("arbor.cost.usd", total_cost)

    except Exception as e:
        print(f"Error logging metrics: {e}")

def check_jailbreak(text):
    """
    Security Layer: Checks for prompt injection attempts.
    """
    forbidden = ["ignore your instructions", "system override", "ignore previous instructions"]
    text_lower = text.lower()

    for phrase in forbidden:
        if phrase in text_lower:
            from ddtrace import tracer
            span = tracer.current_span()
            if span:
                span.set_tag("error", "true")
                span.set_tag("error.message", "Prompt Injection Attempt")
                span.set_metric("arbor.security.jailbreak_attempt", 1)
            return True
    return False

def get_active_lineage(branch_name):
    """
    Recursively fetch history from leaf up to ROOT.
    Returns: [Root History, ..., Parent History, Current Branch History]
    """
    history = []
    current = branch_name
    visited = set() # Cycle detection

    # Traverse up the tree
    while current and current in st.session_state.nodes:
        if current in visited: break # Stop if cycle detected
        visited.add(current)

        node_data = st.session_state.nodes[current]
        # Prepend current node's history (so ROOT is first)
        history = node_data.get("history", []) + history
        current = node_data.get("parent")

    return history

@llm_task(name="fact_extractor")
def extract_global_facts(user_input):
    """
    Background task: Scans input for permanent user facts.
    """
    # Security Check
    if check_jailbreak(user_input):
        return []

    prompt = f"""
    Analyze this text: "{user_input}"
    Extract any permanent facts about the USER (Name, Age, Job, Skills, Location).
    Return ONLY a JSON list of short strings. If none, return [].
    Example: ["User is 22", "User knows Python"]
    """
    try:
        response_obj = model.generate_content(prompt)

        log_execution_metrics(response_obj)
        response_text = response_obj.text

        clean_json = response_text.replace("```json", "").replace("```", "").strip()
        facts = json.loads(clean_json)

        if facts:
            from ddtrace import tracer
            tracer.current_span().set_metric("arbor.facts_learned", len(facts))
        return facts
    except:
        return []

@llm_task(name="router_decision")
def route_topic(user_input, current_branch, all_branches):
    """
    ARBOR 4.2: Tiered Confidence (The "Self-Aware" Router)
    """
    if check_jailbreak(user_input): return "STAY"

    # 1. Embed User Input
    input_vec = get_batch_embeddings([user_input], task_type="RETRIEVAL_QUERY")[0]

    # ---------------------------------------------------------
    # STEP A: CALCULATE "STAY" SCORE (Current Branch)
    # ---------------------------------------------------------
    current_vec = st.session_state.nodes[current_branch].get("vector")
    raw_relevance = cosine_similarity(input_vec, current_vec)

    # Inertia: Only boost if decent match (>0.45)
    # REDUCED BOOST: 0.10 -> 0.02 to allow "Drilling Down" (Child Nodes) more easily.
    if current_branch not in ["ROOT", "Start"] and raw_relevance > 0.45:
        stay_score = min(1.0, raw_relevance + 0.02)
    else:
        stay_score = raw_relevance

    # ---------------------------------------------------------
    # STEP B: CALCULATE "SWITCH" SCORE (Best Sibling)
    # ---------------------------------------------------------
    best_switch_branch = None
    best_switch_score = -1.0

    for branch in all_branches:
        if branch == "ROOT" or branch == current_branch: continue
        branch_vec = st.session_state.nodes[branch].get("vector")
        if branch_vec is None: continue
        score = cosine_similarity(input_vec, branch_vec)
        if score > best_switch_score:
            best_switch_score = score
            best_switch_branch = branch

    # ---------------------------------------------------------
    # STEP C: METRICS (Adaptive Momentum)
    # ---------------------------------------------------------
    drift = 1.0 - stay_score
    prev_drift = st.session_state.get("last_drift_score", 0.0)

    # Fast drop (0.7) if confused, Fast recovery (0.5) if stabilizing
    alpha = 0.7 if drift > prev_drift else 0.5
    smoothed_drift = (prev_drift * (1 - alpha)) + (drift * alpha)
    st.session_state.last_drift_score = smoothed_drift

    from ddtrace import tracer
    tracer.current_span().set_metric("arbor.drift_score", smoothed_drift)

    # ---------------------------------------------------------
    # STEP D: DECISION SHOWDOWN
    # ---------------------------------------------------------

    # 1. SWITCH CHECK
    # Only switch if significantly better (+0.05)
    # LOWERED THRESHOLD: 0.65 -> 0.60 to catch more "return to topic" cases
    if best_switch_score > 0.60 and best_switch_score > (stay_score + 0.05):
        return f"SWITCH:{best_switch_branch}"

    # 2. STAY CHECK (With Tiered Confidence)
    if stay_score >= 0.70:
        # UX FIX: Honest Feedback
        # Instead of lying and saying "95%" for everything, we map reality.

        if stay_score > 0.80:
            # PERFECT MATCH: User is exactly on topic.
            st.session_state.last_drift_score = 0.05  # 95% (Green Lock 🔒)

        else:
            # GOOD MATCH: Standard conversation flow.
            st.session_state.last_drift_score = 0.15  # 85% (Green/Solid ✅)

        return "STAY"

    # 3. CREATE CHECK
    if not os.getenv("PROJECT_ID"):
        new_name = "New Topic"
    else:
        try:
            # Dynamic Naming Strategy based on similarity to current context
            if raw_relevance > 0.40:
                 # Likely a child/related node -> Be Specific
                 name_prompt = f"Name this specific sub-topic in 2-3 words (e.g. 'Making Sandwiches'). Return ONLY the name. No markdown. No punctuation. Input: '{user_input}'"
            else:
                 # Likely a new root topic -> Be Generic
                 name_prompt = f"Classify this input into a BROAD, GENERIC category (1-2 words, e.g. 'Cooking'). Return ONLY the name. No markdown. No punctuation. Input: '{user_input}'"

            raw_name = model.generate_content(name_prompt).text.strip()
            
            # CLEANUP: Remove markdown, quotes, and extra text
            new_name = raw_name.replace("**", "").replace('"', "").replace("'", "").split("\n")[0]
            
            # SAFETY: If model is chatty and returns a sentence, truncate or fallback
            if len(new_name) > 25:
                new_name = new_name[:25] + "..."
                
        except Exception as e:
            print(f"Naming Error: {e}")
            new_name = "New Topic"

    return f"CREATE:{new_name}"

@llm_task(name="generate_response")
def get_ai_response(user_input, global_context, lineage_history):
    """
    Generates answer using Lineage History + Global Root Facts.
    """
    # Security Check
    if check_jailbreak(user_input):
        return "I cannot ignore my core instructions."

    context_str = "\n".join(global_context)
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in lineage_history[-10:]])

    prompt = f"""
    SYSTEM: You are Arbor, a memory-augmented AI.
    GLOBAL CONTEXT (User Facts): {context_str}
    CURRENT CONVERSATION: {history_str}
    User: {user_input}
    Answer naturally.
    """
    response_obj = model.generate_content(prompt)

    log_execution_metrics(response_obj)
    return response_obj.text

# ---------------------------------------------------------
# 3. UI LAYER (Streamlit)
# ---------------------------------------------------------

st.set_page_config(layout="wide", page_title="Arbor")

# Session State Initialization
if "nodes" not in st.session_state:
    # Generate initial index for default nodes
    root_vec = get_batch_embeddings(["ROOT"], task_type="RETRIEVAL_DOCUMENT")[0]
    # Improve Start vector to catch greetings and prevent "Conversation" nodes
    start_vec = get_batch_embeddings(["Start conversation greetings hello hi what's up"], task_type="RETRIEVAL_DOCUMENT")[0]

    st.session_state.nodes = {
        "ROOT": {
            "facts": ["User Session Started"],
            "history": [],
            "parent": None,
            "vector": root_vec  # <--- INDEXED!
        },
        "Start": {
            "history": [],
            "parent": "ROOT",
            "vector": start_vec # <--- INDEXED!
        }
    }
    st.session_state.current_branch = "Start"

# Initialize Metrics in Session State
if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "last_drift_score" not in st.session_state:
    st.session_state.last_drift_score = 0.1 # Default safe
if "tokens_saved" not in st.session_state:
    st.session_state.tokens_saved = 0
if "show_graph_fullscreen" not in st.session_state:
    st.session_state.show_graph_fullscreen = False

# Determine Active Path for Visualization
active_path = set()
curr = st.session_state.current_branch
visited_path = set() # Cycle detection

while curr and curr in st.session_state.nodes:
    if curr in visited_path: break # Stop if cycle detected
    visited_path.add(curr)
    
    active_path.add(curr)
    curr = st.session_state.nodes[curr].get("parent")


# Helper: render interactive graph (agraph when available, graphviz fallback)
def render_graph(fullscreen=False, show_caption=True):
    if agraph and Config:
        nodes = []
        edges = []
        for name, node in st.session_state.nodes.items():
            # Purple/gray theme; labels drawn inside nodes
            color = "#8b5cf6" if name in active_path else "#4b5563"
            font_color = "#ffffff"
            nodes.append(Node(
                id=name,
                label=name,
                size=28 if fullscreen else 22,
                color=color,
                font={"color": font_color, "size": 12, "multi": True}
            ))
            parent = node.get("parent")
            if parent:
                edge_color = "#a78bfa" if (name in active_path and parent in active_path) else "#6b7280"
                edges.append(Edge(source=parent, target=name, color=edge_color, width=2))

        config = Config(
            width=1400 if fullscreen else 320,
            height=850 if fullscreen else 260,
            directed=True,
            physics=True,
            hierarchical=False,
            nodeHighlightBehavior=True,
            highlightColor="#c4b5fd",
            collapsible=False,
            node={'labelProperty': 'label', 'renderLabel': True},
            link={'labelProperty': 'label', 'renderLabel': False},
            d3={'linkLength': 200 if fullscreen else 120, 'charge': -600 if fullscreen else -450},
            staticGraph=False,
            staticGraphWithDragAndDrop=False
        )

        # Dark grid background
        st.markdown("""
        <style>
        .stApp iframe {
            background:
                linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px),
                #0a0a0f !important;
            background-size: 20px 20px !important;
        }
        </style>
        """, unsafe_allow_html=True)

        if show_caption:
            st.caption("Interactive Graph")
        agraph(nodes=nodes, edges=edges, config=config)
    else:
        graph = graphviz.Digraph()
        graph.attr(rankdir='TB', size='2,2', margin='0.1', fontsize='10')
        graph.attr('node', fontsize='10', height='0.3', width='0.5', margin='0.05')
        root_attrs = {"color": "green", "fillcolor": "#e6ffe6", "penwidth": "2"} if "ROOT" in active_path else {"color": "grey", "fillcolor": "#f0f0ff", "penwidth": "1"}
        graph.node("ROOT", label="ROOT", shape="box", style="filled", **root_attrs)
        for name in st.session_state.nodes:
            if name == "ROOT": continue
            if name in active_path:
                color, fill, penwidth = "green", "#e6ffe6", "2"
            else:
                color, fill, penwidth = "grey", "white", "1"
            graph.node(name, label=name, color=color, style="filled", fillcolor=fill, penwidth=penwidth)
            parent = st.session_state.nodes[name].get("parent")
            if parent:
                edge_color = "green" if (name in active_path and parent in active_path) else "grey"
                graph.edge(parent, name, color=edge_color)
        st.graphviz_chart(graph, use_container_width=True)

# Sidebar
with st.sidebar:
    # 1) Context Switch
    all_branches = [n for n in st.session_state.nodes.keys() if n != "ROOT"]
    if st.session_state.current_branch == "ROOT":
        st.session_state.current_branch = "Start"
        st.rerun()

    selected_branch = st.selectbox(
        "Context Switch",
        all_branches,
        index=all_branches.index(st.session_state.current_branch)
    )
    if selected_branch != st.session_state.current_branch:
        st.session_state.current_branch = selected_branch
        st.toast(f"Switched to: {selected_branch}")
        st.rerun()

    # 2) Interactive Graph (sidebar) with hover zoom icon
    zoom_col1, zoom_col2 = st.columns([0.8, 0.2])
    with zoom_col2:
        if st.button("🔍", help="Fullscreen graph"):
            st.session_state.show_graph_fullscreen = True
            st.rerun()

    render_graph(fullscreen=False, show_caption=True)

    # 3) Context Stability Bar
    st.divider()
    drift = st.session_state.get("last_drift_score", 0.0)
    health = max(0, min(100, int((1 - drift) * 100)))
    st.caption(f"Context Stability: {health}%")
    st.progress(health / 100)

    # 4) Compact Metrics
    total_used = st.session_state.total_tokens
    total_saved = st.session_state.get('tokens_saved', 0)
    
    if (total_used + total_saved) > 0:
        efficiency = (total_saved / (total_used + total_saved)) * 100
    else:
        efficiency = 0.0

    c1, c2 = st.columns(2)
    c1.metric("Tokens Used", f"{total_used:,}")
    c2.metric("Tokens Saved", f"{total_saved:,}")
    
    c3, c4 = st.columns(2)
    c3.metric("Est. Cost", f"${st.session_state.total_cost:.4f}")
    c4.metric("Efficiency", f"{efficiency:.1f}%")

# Chat Interface
if st.session_state.get("show_graph_fullscreen"):
    # Fullscreen Graph View
    close_col1, close_col2 = st.columns([0.9, 0.1])
    with close_col1:
        st.subheader("Knowledge Graph")
    with close_col2:
        if st.button("Close ✖", use_container_width=True):
            st.session_state.show_graph_fullscreen = False
            st.rerun()
            
    render_graph(fullscreen=True, show_caption=False)
    st.stop() # Halt execution to hide chat interface

st.title("Arbor")
st.caption(f"Active Context: **{st.session_state.current_branch}**")

# Render Local History for Display
for msg in st.session_state.nodes[st.session_state.current_branch]["history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Feedback Callback
def handle_feedback():
    if "feedback_key" in st.session_state and st.session_state.feedback_key:
        fb = st.session_state.feedback_key
        score = 0.0
        if fb == 1: # Thumbs Up
            score = 1.0
            metric_name = "arbor.feedback.positive"
        else: # Thumbs Down (0)
            score = 0.0
            metric_name = "arbor.feedback.negative"

        from ddtrace import tracer
        
        # FIX: Use standard tracer instead of LLMObs.annotate which requires an active LLM span
        # This avoids the "No active LLMObs-generated span found" error while still tracking the data.
        with tracer.trace("arbor.user_feedback") as span:
            span.set_metric(metric_name, 1)
            span.set_tag("evaluation.quality", score)

        st.toast("Thanks for your feedback!")

# Input Loop
if prompt := st.chat_input("What's on your mind?"):
    with st.chat_message("user"):
        st.write(prompt)

    # 1. Background: Learn Facts
    new_facts = extract_global_facts(prompt)
    if new_facts:
        # DEDUPLICATION: Prevent storing the same fact twice
        existing_facts = set(st.session_state.nodes["ROOT"]["facts"])
        unique_new_facts = [f for f in new_facts if f not in existing_facts]

        if unique_new_facts:
            st.session_state.nodes["ROOT"]["facts"].extend(unique_new_facts)
            st.toast(f"Memorized: {unique_new_facts}", icon="💾")

    # 2. Decision: Route & Self-Heal
    decision = route_topic(prompt, st.session_state.current_branch, list(st.session_state.nodes.keys()))

    target_branch = st.session_state.current_branch

    if "SWITCH:" in decision:
        target_branch = decision.split(":", 1)[1]
        if target_branch == "ROOT":
             st.toast("⚠️ High Context Drift Detected! Resetting to Root...", icon="🚨")
        else:
             st.info(f"Switching to: {target_branch}")
        
        # FIX: Reset drift score because we found a good match
        st.session_state.last_drift_score = 0.1

    elif "CREATE:" in decision:
        new_name = decision.split(":", 1)[1].strip()
        
        # FALLBACK: Ensure name is not empty
        if not new_name:
            new_name = "New Topic"

        # PROTECTION: Prevent overwriting existing nodes
        original_name = new_name
        counter = 2
        while new_name in st.session_state.nodes:
            new_name = f"{original_name} ({counter})"
            counter += 1

        # ---------------------------------------------------------
        # PARENTING LOGIC: Hybrid Search (Ancestry + Global)
        # ---------------------------------------------------------
        # 1. Local Context (Ancestry): Prefer keeping context if relevant.
        # 2. Global Context: If unrelated to current chain, look elsewhere.
        # 3. Fallback: If nothing matches, it's a new root topic.

        input_vec = get_batch_embeddings([prompt], task_type="RETRIEVAL_QUERY")[0]
        
        # A. Find Best Ancestor
        best_ancestor = None
        best_anc_score = -1.0
        
        curr = st.session_state.current_branch
        while curr and curr in st.session_state.nodes:
            if curr == "ROOT": break # Skip ROOT
            
            vec = st.session_state.nodes[curr].get("vector")
            score = cosine_similarity(input_vec, vec)
            
            if score > best_anc_score:
                best_anc_score = score
                best_ancestor = curr
            
            curr = st.session_state.nodes[curr].get("parent")

        # B. Find Best Global (if Ancestor is weak)
        best_global = None
        best_global_score = -1.0
        
        # Only scan global if local is not a "slam dunk" (>0.75)
        if best_anc_score < 0.75:
            for name, node in st.session_state.nodes.items():
                if name in ["ROOT", "Start", new_name]: continue
                
                score = cosine_similarity(input_vec, node["vector"])
                if score > best_global_score:
                    best_global_score = score
                    best_global = name

        # C. Decision Logic
        # Thresholds
        STRONG_MATCH = 0.72  # High bar for automatic acceptance
        WEAK_MATCH = 0.58    # Lower bar requires LLM verification
        BETTER_MATCH_MARGIN = 0.10
        
        parent_node = "Start" # Default
        
        # 1. Prefer Ancestor if it's strong
        if best_ancestor and best_anc_score > STRONG_MATCH:
            parent_node = best_ancestor
            st.toast(f"Kept Context: {parent_node}", icon="🔗")
            
        # 2. Switch to Global if it's significantly better
        elif best_global and best_global_score > STRONG_MATCH and best_global_score > (best_anc_score + BETTER_MATCH_MARGIN):
            parent_node = best_global
            st.toast(f"Re-routed to: {parent_node}", icon="twisted_rightwards_arrows")
            
        # 3. Gray Zone (0.58 - 0.72): Ask LLM for a "Vibe Check"
        elif best_ancestor and best_anc_score > WEAK_MATCH:
             try:
                 check_prompt = f"""
                 Is the new topic '{new_name}' a sub-topic or directly related to '{best_ancestor}'?
                 Answer YES or NO.
                 """
                 check_resp = model.generate_content(check_prompt).text.strip().upper()
                 
                 if "YES" in check_resp:
                     parent_node = best_ancestor
                     st.toast(f"Verified Context: {parent_node}", icon="✅")
                 else:
                     st.toast(f"Context Rejected: {best_ancestor}", icon="🚫")
             except:
                 # On error, be conservative and split
                 pass
             
        # 4. Else: New Topic (Start)
        else:
             st.toast(f"New Topic Created: {new_name}", icon="🌿")

        # INDEXING STEP
        # Path-Awareness: Inherit parent name for better vector search later
        if parent_node and parent_node not in ["ROOT", "Start"]:
            contextual_text = f"{parent_node} {new_name}"
        else:
            contextual_text = new_name

        new_vec = get_batch_embeddings([contextual_text], task_type="RETRIEVAL_DOCUMENT")[0]

        st.session_state.nodes[new_name] = {
            "history": [],
            "parent": parent_node,
            "vector": new_vec
        }
        target_branch = new_name

        # Reset relevance (drift score) for the new node
        st.session_state.last_drift_score = 0.1

    st.session_state.current_branch = target_branch

    # 3. Respond
    with st.chat_message("assistant"):
        # --- TRUST BADGE ---
        drift = st.session_state.get("last_drift_score", 0.1)
        if drift < 0.2:
            st.markdown("🛡️ **Verified Context**", unsafe_allow_html=True)
        elif drift <= 0.8:
            st.markdown("🤔 **Context Bridging**", unsafe_allow_html=True)
        else:
            st.markdown("⚠️ **Context Drift Detected**", unsafe_allow_html=True)

        # Arbor 2.0: Get active lineage history
        lineage_history = get_active_lineage(target_branch)
        resp = get_ai_response(prompt, st.session_state.nodes["ROOT"]["facts"], lineage_history)
        st.write(resp)

    st.session_state.nodes[target_branch]["history"].append({"role": "user", "content": prompt})
    st.session_state.nodes[target_branch]["history"].append({"role": "assistant", "content": resp})

    st.rerun()

# Show feedback widget for the LAST message if it was from assistant
current_hist = st.session_state.nodes[st.session_state.current_branch]["history"]
if current_hist and current_hist[-1]["role"] == "assistant":
    st.feedback("thumbs", key="feedback_key", on_change=handle_feedback)
