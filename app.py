import streamlit as st
import graphviz
import json
import os
from dotenv import load_dotenv

# 1. SETUP & CONFIG
load_dotenv()

# Detect Mock Mode
PROJECT_ID = os.getenv("PROJECT_ID")
DD_API_KEY = os.getenv("DD_API_KEY")
MOCK_MODE = not PROJECT_ID or not DD_API_KEY or "your_" in PROJECT_ID or "your_" in DD_API_KEY

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
    model = GenerativeModel("gemini-1.5-flash-001")
else:
    print("WARNING: Running in MOCK MODE due to missing credentials.")

# ---------------------------------------------------------
# 2. INTELLIGENCE LAYERS (The "Brain")
# ---------------------------------------------------------

class MockGenerativeModel:
    def generate_content(self, prompt):
        class MockResponse:
            def __init__(self, text):
                self.text = text

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
            return LLMObs.task(name=name)(func)
        return func
    return decorator

def get_lineage_history(leaf_branch):
    """
    Recursively fetch history from leaf up to ROOT.
    """
    history = []
    current = leaf_branch

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
    Background task: Scans input for permanent user facts (Name, Job, Skills)
    to store in the 'Root' node for global context.
    """
    prompt = f"""
    Analyze this text: "{user_input}"
    Extract any permanent facts about the USER (Name, Age, Job, Skills, Location).
    Return ONLY a JSON list of short strings. If none, return [].
    Example: ["User is 22", "User knows Python"]
    """
    try:
        if MOCK_MODE:
            response = MockGenerativeModel().generate_content(prompt).text
        else:
            response = model.generate_content(prompt).text

        clean_json = response.replace("```json", "").replace("```", "").strip()
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
    if MOCK_MODE:
        response = MockGenerativeModel().generate_content(prompt).text.strip()
    else:
        response = model.generate_content(prompt).text.strip()

    # Metrics for Hackathon "Wow" Factor
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
    context_str = "\n".join(global_context)
    # Use the full lineage history instead of just the branch history
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in lineage_history[-10:]])

    prompt = f"""
    SYSTEM: You are Arbor, a memory-augmented AI.
    GLOBAL CONTEXT (User Facts): {context_str}
    CURRENT CONVERSATION: {history_str}
    User: {user_input}
    Answer naturally.
    """
    if MOCK_MODE:
        return MockGenerativeModel().generate_content(prompt).text
    return model.generate_content(prompt).text

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
    # Highlight ROOT if it is in the active path (it always should be)
    root_color = "green" if "ROOT" in active_path else "purple" # Keep purple as base or switch to green if active?
    # User asked for: "highlight the entire active path (Root -> Parent -> Child) in green"
    # So ROOT should be green if active. But originally it was purple. Let's make it green if active.
    # Actually, let's keep the shape distinct but change the outline/fill to indicate active path.

    if "ROOT" in active_path:
        root_attrs = {"color": "green", "fillcolor": "#e6ffe6"}
    else:
        root_attrs = {"color": "purple", "fillcolor": "#f0f0ff"}

    graph.node("ROOT", label=f"ROOT\n{root_facts}", shape="doubleoctagon", style="filled", **root_attrs)

    # Dynamic Branches
    for name in st.session_state.nodes:
        if name == "ROOT": continue

        # Determine styling based on active path
        if name in active_path:
            color = "green"
            fill = "#e6ffe6"
        else:
            color = "grey"
            fill = "white"

        graph.node(name, label=name, color=color, style="filled", fillcolor=fill)

        # Edge Logic: Connect to parent
        parent = st.session_state.nodes[name].get("parent")
        if parent:
            edge_color = "green" if (name in active_path and parent in active_path) else "grey"
            graph.edge(parent, name, color=edge_color)

    st.graphviz_chart(graph)

# Chat Interface
st.title("Arbor")
st.caption(f"Active Context: **{st.session_state.current_branch}**")

# Render Conversation (Current Branch Only? Or Lineage?)
# User requirement: "The UI (Streamlit) visualizes the conversation as a tree graph."
# Usually chat interfaces show the history relevant to the current context.
# Let's show the local history of the current branch to keep it clean,
# or the lineage if we want to show the 'context' the AI sees.
# The prompt says: "Render Conversation ... for msg in st.session_state.nodes[st.session_state.current_branch]['history']"
# I will stick to the local history for display as per the original code structure unless asked otherwise.
for msg in st.session_state.nodes[st.session_state.current_branch]["history"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input Loop
if prompt := st.chat_input("What's on your mind?"):
    with st.chat_message("user"):
        st.write(prompt)

    # 1. Background: Learn Facts
    new_facts = extract_global_facts(prompt)
    if new_facts:
        st.session_state.nodes["ROOT"]["facts"].extend(new_facts)
        st.toast(f"Memorized: {new_facts}", icon="💾")

    # 2. Decision: Route
    decision = route_topic(prompt, st.session_state.current_branch, list(st.session_state.nodes.keys()))

    target_branch = st.session_state.current_branch
    if "CREATE:" in decision:
        new_name = decision.split(":")[1]
        # Arbor 2.0: Assign parent as the current branch
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
        # Arbor 2.0: Get lineage history
        lineage_history = get_lineage_history(target_branch)
        resp = get_ai_response(prompt, st.session_state.nodes["ROOT"]["facts"], lineage_history)
        st.write(resp)

    st.session_state.nodes[target_branch]["history"].append({"role": "user", "content": prompt})
    st.session_state.nodes[target_branch]["history"].append({"role": "assistant", "content": resp})
    st.rerun()
