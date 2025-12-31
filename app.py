import streamlit as st
import graphviz
import json
import os
import random
from dotenv import load_dotenv

# 1. SETUP & CONFIG
load_dotenv()

# Detect Mode
PROJECT_ID = os.getenv("PROJECT_ID")
DD_API_KEY = os.getenv("DD_API_KEY")
DD_SITE = os.getenv("DD_SITE")
MODEL = os.getenv("MODEL")

import vertexai
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
import numpy as np
from ddtrace import tracer, patch_all
from ddtrace.llmobs import LLMObs

patch_all()

# Initialize Datadog
if DD_API_KEY:
    LLMObs.enable(
        ml_app=os.getenv("DD_SERVICE"),
        api_key=DD_API_KEY,
        site=os.getenv("DD_SITE")
    )

# Initialize Google Vertex AI
vertexai.init(project=PROJECT_ID, location="us-central1")
model = GenerativeModel(MODEL)

# Load the lightweight embedding model
embedding_model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

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

    # Traverse up the tree
    while current and current in st.session_state.nodes:
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
    ARBOR 4.0: Indexed Routing (Maximum Efficiency).
    """
    if check_jailbreak(user_input): return "STAY"

    # 1. Embed User Input (The ONLY API Call we make now!)
    input_vec = get_batch_embeddings([user_input], task_type="RETRIEVAL_QUERY")[0]

    # 2. Retrieve Cached Vectors (Zero Cost)
    # Get current branch vector from memory
    current_vec = st.session_state.nodes[current_branch].get("vector")

    # 3. Calculate Drift (Math only)
    raw_relevance = cosine_similarity(input_vec, current_vec)

    # ---------------------------------------------------------
    # CONDITIONAL INERTIA (The Fix)
    # Only give a boost if the topic is ALREADY somewhat related (> 0.45).
    # This helps "Tent" (0.55 -> 0.65) stick,
    # but stops "Python" (0.20) from getting a free pass.
    # ---------------------------------------------------------
    if current_branch not in ["ROOT", "Start"] and raw_relevance > 0.45:
        relevance = min(1.0, raw_relevance + 0.10) # Lower boost to +0.10
    else:
        relevance = raw_relevance

    instant_drift = 1.0 - relevance

    # Apply Momentum (EMA) for UI Stability
    prev_drift = st.session_state.get("last_drift_score", 0.0)
    alpha = 0.3
    smoothed_drift = (prev_drift * (1 - alpha)) + (instant_drift * alpha)

    st.session_state.last_drift_score = smoothed_drift

    # Log to Datadog
    from ddtrace import tracer
    tracer.current_span().set_metric("arbor.drift_score", smoothed_drift)

    # 4. DECISION LOGIC (Stricter Thresholds)

    # STRICTER STAY THRESHOLD: Raised to 0.60
    if relevance > 0.60:
        return "STAY"

    # Search for Switches
    best_match = None
    best_score = -1.0

    for branch in all_branches:
        if branch == "ROOT" or branch == current_branch: continue
        branch_vec = st.session_state.nodes[branch].get("vector")
        if branch_vec is None: continue

        score = cosine_similarity(input_vec, branch_vec)
        if score > best_score:
            best_score = score
            best_match = branch

    # STRICTER SWITCH THRESHOLD: Raised to 0.65
    if best_score > 0.65:
        return f"SWITCH:{best_match}"

    # If we are here, it's definitely a NEW topic.
    # Generate a name.
    name_prompt = f"Name the topic of this input in 2-3 words: '{user_input}'"
    new_name = model.generate_content(name_prompt).text.strip()

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
    start_vec = get_batch_embeddings(["Start"], task_type="RETRIEVAL_DOCUMENT")[0]

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

# Determine Active Path for Visualization
active_path = set()
curr = st.session_state.current_branch
while curr:
    active_path.add(curr)
    curr = st.session_state.nodes[curr].get("parent")

# Sidebar: Visual Tree
with st.sidebar:
    st.header("🧠 Memory Topology")

    st.header("🕰️ Context Time Travel")

    # Get all available branches
    all_branches = list(st.session_state.nodes.keys())

    # Create a Selectbox to manually jump branches
    # Default to current_branch
    selected_branch = st.selectbox(
        "Jump to Topic:",
        all_branches,
        index=all_branches.index(st.session_state.current_branch)
    )

    # Logic: If user changes the dropdown, FORCE a switch
    if selected_branch != st.session_state.current_branch:
        st.session_state.current_branch = selected_branch
        st.toast(f"⏳ Time Travelled to: {selected_branch}", icon="🚀")
        st.rerun()

    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')

    # Root Node
    root_facts = "<br/>".join(st.session_state.nodes["ROOT"]["facts"])

    if "ROOT" in active_path:
        root_attrs = {"color": "green", "fillcolor": "#e6ffe6", "penwidth": "2"}
    else:
        root_attrs = {"color": "grey", "fillcolor": "#f0f0ff", "penwidth": "1"}

    graph.node("ROOT", label=f"ROOT\n{root_facts}", shape="doubleoctagon", style="filled", **root_attrs)

    # Dynamic Branches
    for name in st.session_state.nodes:
        if name == "ROOT": continue

        # Determine styling based on active path
        if name in active_path:
            color = "green"
            fill = "#e6ffe6"
            penwidth = "2"
        else:
            color = "grey"
            fill = "white"
            penwidth = "1"

        graph.node(name, label=name, color=color, style="filled", fillcolor=fill, penwidth=penwidth)

        # Edge Logic: Connect to parent
        parent = st.session_state.nodes[name].get("parent")
        if parent:
            edge_color = "green" if (name in active_path and parent in active_path) else "grey"
            edge_width = "2" if (name in active_path and parent in active_path) else "1"
            graph.edge(parent, name, color=edge_color, penwidth=edge_width)

    st.graphviz_chart(graph)

    # --- FINOPS & HEALTH DASHBOARD ---
    st.divider()
    st.subheader("📊 System Health")

    # Context Health Bar
    drift = st.session_state.get("last_drift_score", 0.0)
    health = max(0, min(100, int((1 - drift) * 100)))
    st.caption(f"Context Stability: {health}%")

    # Dynamic Color for Bar
    # Streamlit progress doesn't natively support color arg in all versions,
    # but let's stick to standard progress.
    st.progress(health / 100)

    st.subheader("💰 Efficiency Metrics")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Tokens Used", f"{st.session_state.total_tokens}")
    with c2:
        st.metric("Est. Cost", f"${st.session_state.total_cost:.4f}")

    # The 'Mic Drop' Metric
    st.metric("🚫 Tokens Saved (Arbor Optimization)",
              f"{st.session_state.get('tokens_saved', 0)}",
              delta="Efficiency +40%")

# Chat Interface
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
        from ddtrace.llmobs import LLMObs
        span = tracer.trace("arbor.user_feedback")
        span.set_metric(metric_name, 1)
        LLMObs.annotate(tags={"evaluation.quality": score})
        span.finish()

        st.toast("Thanks for your feedback!")

# Input Loop
if prompt := st.chat_input("What's on your mind?"):
    with st.chat_message("user"):
        st.write(prompt)

    # 1. Background: Learn Facts
    new_facts = extract_global_facts(prompt)
    if new_facts:
        st.session_state.nodes["ROOT"]["facts"].extend(new_facts)
        st.toast(f"Memorized: {new_facts}", icon="💾")

    # 2. Decision: Route & Self-Heal
    decision = route_topic(prompt, st.session_state.current_branch, list(st.session_state.nodes.keys()))

    target_branch = st.session_state.current_branch

    if "SWITCH:" in decision:
        target_branch = decision.split(":")[1]
        if target_branch == "ROOT":
             st.toast("⚠️ High Context Drift Detected! Resetting to Root...", icon="🚨")
        else:
             st.info(f"Switching to: {target_branch}")

    elif "CREATE:" in decision:
        new_name = decision.split(":")[1]

        # Calculate Relevance from Drift (Relevance = (1-Drift)*10)
        drift = st.session_state.get("last_drift_score", 0.0)
        relevance = (1.0 - drift) * 10.0

        # DEFAULT DECISION: Based on Thresholds
        is_child = False

        # Case 1: Clear Match (High Trust)
        if relevance > 6.0:
            is_child = True

        # Case 2: The "Gray Zone" (Ambiguous)
        # Score is between 5.0 and 6.0 (e.g., your 59% Resume/Intern case)
        elif relevance > 5.0:
            # TIE-BREAKER: Ask the "Brain" (LLM) for a second opinion
            # This is slower but much smarter than a raw vector check.
            check_prompt = f"""
            Task: Parenting Check.
            Is the new topic '{new_name}' a direct sub-step or detail of '{st.session_state.current_branch}'?
            Context: The user is switching from '{st.session_state.current_branch}' to '{new_name}'.
            Answer YES or NO only.
            """
            try:
                # Quick call to Gemini
                check_resp = model.generate_content(check_prompt).text.strip().upper()
                if "YES" in check_resp:
                    is_child = True
                    st.toast("🧠 AI Tie-Breaker: Connected related topics!", icon="🔗")
            except:
                is_child = False

        # EXECUTE DECISION
        if is_child:
            parent_node = st.session_state.current_branch
            st.success(f"Drilling down: {st.session_state.current_branch} → {new_name}")
        else:
            # Sibling Logic
            if st.session_state.current_branch in ["ROOT", "Start"]:
                 parent_node = st.session_state.current_branch
            else:
                 parent_node = "Start"

            st.toast(f"New Branch Created: {new_name}", icon="🌿")

        # INDEXING STEP: Generate vector for the new name immediately
        # PATH-AWARENESS LOGIC:
        # Instead of embedding just "Sauce", we embed "Cooking > Pasta > Sauce".
        # This ensures the child vector inherits the semantic context of the parent.

        if parent_node and parent_node not in ["ROOT", "Start"]:
            contextual_text = f"{parent_node} {new_name}"
        else:
            contextual_text = new_name

        new_vec = get_batch_embeddings([contextual_text], task_type="RETRIEVAL_DOCUMENT")[0]

        st.session_state.nodes[new_name] = {
            "history": [],
            "parent": parent_node,
            "vector": new_vec  # <--- Store it for future lookups
        }
        target_branch = new_name

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
