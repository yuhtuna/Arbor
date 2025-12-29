import streamlit as st
import graphviz
import json
import os
import random
from dotenv import load_dotenv

# 1. SETUP & CONFIG
load_dotenv()

# Detect Mock Mode
PROJECT_ID = os.getenv("PROJECT_ID")
DD_API_KEY = os.getenv("DD_API_KEY")
MOCK_MODE = not PROJECT_ID or not DD_API_KEY or "your_" in PROJECT_ID or "your_" in DD_API_KEY
MODEL = os.getenv("MODEL")
if not MOCK_MODE:
    import vertexai
    from vertexai.generative_models import GenerativeModel
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
else:
    print("WARNING: Running in MOCK MODE due to missing credentials.")

# ---------------------------------------------------------
# 2. INTELLIGENCE LAYERS (The "Brain")
# ---------------------------------------------------------

class MockGenerativeModel:
    def generate_content(self, prompt):
        class MockUsage:
            prompt_token_count = 50
            candidates_token_count = 20

        class MockResponse:
            def __init__(self, text):
                self.text = text
                self.usage_metadata = MockUsage()

        prompt_str = str(prompt)

        # 1. Fact Extraction Mock
        if "Extract any permanent facts" in prompt_str:
            if "Python" in prompt_str:
                return MockResponse('["User knows Python"]')
            elif "cook" in prompt_str or "pasta" in prompt_str:
                return MockResponse('["User likes cooking"]')
            return MockResponse('[]')

        # 2. Router Mock
        if "Task: Route this input" in prompt_str:
            # Self-Healing Trigger
            if "drift" in prompt_str:
                return MockResponse("DRIFT_RESET")

            if "cook" in prompt_str or "pasta" in prompt_str:
                return MockResponse("CREATE:Cooking")
            elif "learn" in prompt_str or "Python" in prompt_str:
                return MockResponse("STAY")
            return MockResponse("STAY")

        # 3. Chat Response Mock
        return MockResponse("This is a mock response from Arbor (Mock Mode). I can see your input and I am updating the tree!")

# Wrapper to handle LLM tasks conditionally
def llm_task(name):
    def decorator(func):
        if not MOCK_MODE:
            from ddtrace.llmobs import LLMObs
            def wrapper(*args, **kwargs):
                with LLMObs.task(name=name):
                    return func(*args, **kwargs)
            return wrapper
        return func
    return decorator

# --- OBSERVABILITY HELPERS ---

def log_execution_metrics(response):
    """
    Extracts tokens and calculates cost for Datadog.
    Pricing (Gemini Flash Approx):
    Input: ~$0.00001875 per 1k characters
    Output: ~$0.000075 per 1k characters
    We estimate 1 token ~= 4 chars for cost calc.
    """
    if MOCK_MODE:
        # We can still simulate the metrics logging in logs if we wanted,
        # but the tracer isn't available.
        return

    from ddtrace import tracer

    try:
        usage = response.usage_metadata
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        total_tokens = input_tokens + output_tokens

        # Calculate Cost (Approximate)
        # 1 token approx 4 characters
        input_cost = (input_tokens * 4 / 1000) * 0.00001875
        output_cost = (output_tokens * 4 / 1000) * 0.000075
        total_cost = input_cost + output_cost

        span = tracer.current_span()
        if span:
            span.set_metric("arbor.tokens.input", input_tokens)
            span.set_metric("arbor.tokens.output", output_tokens)
            span.set_metric("arbor.tokens.total", total_tokens)
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
            if not MOCK_MODE:
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
        if MOCK_MODE:
            response_obj = MockGenerativeModel().generate_content(prompt)
        else:
            response_obj = model.generate_content(prompt)

        log_execution_metrics(response_obj)
        response_text = response_obj.text

        clean_json = response_text.replace("```json", "").replace("```", "").strip()
        facts = json.loads(clean_json)

        if facts and not MOCK_MODE:
            from ddtrace import tracer
            tracer.current_span().set_metric("arbor.facts_learned", len(facts))
        return facts
    except:
        return []

@llm_task(name="router_decision")
def route_topic(user_input, current_branch, all_branches):
    """
    Core Logic: Decides if we STAY, SWITCH branch, or CREATE new branch.
    """
    # Security Check
    if check_jailbreak(user_input):
        return "STAY" # Or handle specifically, but staying is safe.

    prompt = f"""
    Current Active Branch: '{current_branch}'
    Existing Branches: {all_branches}
    User Input: "{user_input}"

    Task: Route this input.
    1. If it fits the Current Branch goal -> Return "STAY"
    2. If it fits an Existing Branch -> Return "SWITCH:BranchName"
    3. If it is a new topic -> Return "CREATE:NewBranchName" (2-3 words max)

    Return ONLY the string decision.
    """

    drift_score = 0.1 # Default low drift

    if MOCK_MODE:
        response_obj = MockGenerativeModel().generate_content(prompt)
        log_execution_metrics(response_obj)
        response = response_obj.text.strip()
        if response == "DRIFT_RESET":
            drift_score = 0.9
    else:
        response_obj = model.generate_content(prompt)
        log_execution_metrics(response_obj)
        response = response_obj.text.strip()
        if "RESET" in response:
             drift_score = 0.9

    # Datadog Metric for Drift
    if not MOCK_MODE:
        from ddtrace import tracer
        tracer.current_span().set_metric("arbor.drift_score", drift_score)

    # Self-Healing Logic
    if drift_score > 0.8:
        return "RESET"

    # Standard Routing Metrics
    if not MOCK_MODE:
        from ddtrace import tracer
        from ddtrace.llmobs import LLMObs
        if "CREATE" in response:
            tracer.current_span().set_metric("arbor.drift.new_branch", 1)
            LLMObs.annotate(tags={"drift_type": "new_topic"})
        elif "SWITCH" in response:
            tracer.current_span().set_metric("arbor.drift.switch_context", 1)
            LLMObs.annotate(tags={"drift_type": "switch_branch"})

    return response

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
    if MOCK_MODE:
        response_obj = MockGenerativeModel().generate_content(prompt)
    else:
        response_obj = model.generate_content(prompt)

    log_execution_metrics(response_obj)
    return response_obj.text

# ---------------------------------------------------------
# 3. UI LAYER (Streamlit)
# ---------------------------------------------------------

st.set_page_config(layout="wide", page_title="Arbor")

# Session State Initialization
if "nodes" not in st.session_state:
    st.session_state.nodes = {
        "ROOT": {"facts": ["User Session Started"], "history": [], "parent": None},
        "Start": {"history": [], "parent": "ROOT"}
    }
    st.session_state.current_branch = "Start"

# Determine Active Path for Visualization
active_path = set()
curr = st.session_state.current_branch
while curr:
    active_path.add(curr)
    curr = st.session_state.nodes[curr].get("parent")

# Sidebar: Visual Tree
with st.sidebar:
    st.header("🧠 Memory Topology")
    if MOCK_MODE:
        st.warning("⚠️ Running in Mock Mode (No valid API Keys detected)")

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
        # 'thumbs' returns {'score': '👍'/'👎'} or similar depending on version,
        # but st.feedback('thumbs') usually returns 1 (up) or 0 (down) in new versions,
        # or we check documentation. Assuming standard Streamlit:
        # Actually st.feedback is very new (1.37+). It often returns an index or value.

        score = 0.0
        # If it returns 1 for up, 0 for down
        if fb == 1: # Thumbs Up
            score = 1.0
            metric_name = "arbor.feedback.positive"
        else: # Thumbs Down (0)
            score = 0.0
            metric_name = "arbor.feedback.negative"

        if not MOCK_MODE:
            from ddtrace import tracer
            from ddtrace.llmobs import LLMObs
            # We treat feedback as an event
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

    if decision == "RESET":
        # Self-Healing Action
        target_branch = "ROOT"
        st.toast("⚠️ High Context Drift Detected! Resetting to Root...", icon="🚨")
    elif "CREATE:" in decision:
        new_name = decision.split(":")[1]
        st.session_state.nodes[new_name] = {
            "history": [],
            "parent": st.session_state.current_branch
        }
        target_branch = new_name
        st.success(f"Context Drift! Spawning: {new_name}")
    elif "SWITCH:" in decision:
        target_branch = decision.split(":")[1]
        st.info(f"Switching to: {target_branch}")

    st.session_state.current_branch = target_branch

    # 3. Respond
    with st.chat_message("assistant"):
        # Arbor 2.0: Get active lineage history
        lineage_history = get_active_lineage(target_branch)
        resp = get_ai_response(prompt, st.session_state.nodes["ROOT"]["facts"], lineage_history)
        st.write(resp)

    st.session_state.nodes[target_branch]["history"].append({"role": "user", "content": prompt})
    st.session_state.nodes[target_branch]["history"].append({"role": "assistant", "content": resp})

    # Rerun to update state and show feedback for the NEW message
    st.rerun()

# Show feedback widget for the LAST message if it was from assistant
# We place this outside the input loop so it persists
current_hist = st.session_state.nodes[st.session_state.current_branch]["history"]
if current_hist and current_hist[-1]["role"] == "assistant":
    st.feedback("thumbs", key="feedback_key", on_change=handle_feedback)
