import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
from ddtrace import tracer, patch_all
from ddtrace.llmobs import LLMObs
import graphviz
import json
import os
from dotenv import load_dotenv

# 1. SETUP & CONFIG
load_dotenv()
patch_all()

# Initialize Datadog
if os.getenv("DD_API_KEY"):
    LLMObs.enable(
        ml_app=os.getenv("DD_SERVICE"),
        api_key=os.getenv("DD_API_KEY"),
        site=os.getenv("DD_SITE")
    )

# Initialize Google Vertex AI
vertexai.init(project=os.getenv("PROJECT_ID"), location="us-central1")
model = GenerativeModel("gemini-1.5-flash-001")

# ---------------------------------------------------------
# 2. INTELLIGENCE LAYERS (The "Brain")
# ---------------------------------------------------------

@LLMObs.task(name="fact_extractor")
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
        response = model.generate_content(prompt).text
        clean_json = response.replace("```json", "").replace("```", "").strip()
        facts = json.loads(clean_json)
        if facts:
            tracer.current_span().set_metric("arbor.facts_learned", len(facts))
        return facts
    except:
        return []

@LLMObs.task(name="router_decision")
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
    response = model.generate_content(prompt).text.strip()

    # Metrics for Hackathon "Wow" Factor
    if "CREATE" in response:
        tracer.current_span().set_metric("arbor.drift.new_branch", 1)
        LLMObs.annotate(tags={"drift_type": "new_topic"})
    elif "SWITCH" in response:
        tracer.current_span().set_metric("arbor.drift.switch_context", 1)
        LLMObs.annotate(tags={"drift_type": "switch_branch"})

    return response

@LLMObs.task(name="generate_response")
def get_ai_response(user_input, global_context, branch_history):
    """
    Generates answer using Local History + Global Root Facts.
    """
    context_str = "\n".join(global_context)
    history_str = "\n".join([f"{m['role']}: {m['content']}" for m in branch_history[-5:]])

    prompt = f"""
    SYSTEM: You are Arbor, a memory-augmented AI.
    GLOBAL CONTEXT (User Facts): {context_str}
    CURRENT CONVERSATION: {history_str}
    User: {user_input}
    Answer naturally.
    """
    return model.generate_content(prompt).text

# ---------------------------------------------------------
# 3. UI LAYER (Streamlit)
# ---------------------------------------------------------

st.set_page_config(layout="wide", page_title="Arbor")

# Session State Initialization
if "nodes" not in st.session_state:
    st.session_state.nodes = {
        "ROOT": {"facts": ["User Session Started"], "history": []},
        "Start": {"history": []}
    }
    st.session_state.current_branch = "Start"

# Sidebar: Visual Tree
with st.sidebar:
    st.header("🧠 Memory Topology")
    graph = graphviz.Digraph()
    graph.attr(rankdir='TB')

    # Root Node
    root_facts = "<br/>".join(st.session_state.nodes["ROOT"]["facts"])
    graph.node("ROOT", label=f"ROOT\n{root_facts}", shape="doubleoctagon", color="purple", style="filled", fillcolor="#f0f0ff")

    # Dynamic Branches
    for name in st.session_state.nodes:
        if name == "ROOT": continue
        color = "green" if name == st.session_state.current_branch else "grey"
        fill = "#e6ffe6" if name == st.session_state.current_branch else "white"
        graph.node(name, label=name, color=color, style="filled", fillcolor=fill)
        graph.edge("ROOT", name)

    st.graphviz_chart(graph)

# Chat Interface
st.title("Arbor")
st.caption(f"Active Context: **{st.session_state.current_branch}**")

# Render Conversation
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
        st.session_state.nodes[new_name] = {"history": []}
        target_branch = new_name
        st.success(f"Context Drift! Spawning: {new_name}")
    elif "SWITCH:" in decision:
        target_branch = decision.split(":")[1]
        st.info(f"Switching to: {target_branch}")

    st.session_state.current_branch = target_branch

    # 3. Respond
    with st.chat_message("assistant"):
        resp = get_ai_response(prompt, st.session_state.nodes["ROOT"]["facts"], st.session_state.nodes[target_branch]["history"])
        st.write(resp)

    st.session_state.nodes[target_branch]["history"].append({"role": "user", "content": prompt})
    st.session_state.nodes[target_branch]["history"].append({"role": "assistant", "content": resp})
    st.rerun()
