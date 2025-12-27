# Arbor: Hierarchical AI Memory Visualization

Arbor is a Streamlit application that visualizes AI memory as a growing tree to prevent "context drift." It uses Google Vertex AI (Gemini Flash) for intelligence and Datadog LLM Observability for tracing and metrics.

**Arbor 2.0 Features:**
*   **Hierarchical Memory:** Conversations are structured as a tree. Every new topic branches off from its parent context.
*   **Recursive Context:** The AI's context window is constructed dynamically by traversing the "active lineage" (from the current leaf node up to the root), effectively pruning irrelevant sibling branches.
*   **Active Path Visualization:** The UI highlights the entire active conversation path in green, giving users a clear visual map of their current context.

## Project Structure

- `app.py`: The main Streamlit application.
- `requirements.txt`: Python dependencies.
- `.env`: Configuration file for API keys and settings.

## Setup

### 1. System Dependencies

**Important:** This application requires **Graphviz** to be installed on your system (not just the Python library).

- **Mac (Homebrew):**
  ```bash
  brew install graphviz
  ```
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get update
  sudo apt-get install graphviz
  ```
- **Windows:**
  Download and install from the [Graphviz website](https://graphviz.org/download/). Make sure to add Graphviz to your system PATH.

### 2. Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Configuration

Open the `.env` file and update the following values with your specific credentials.

**For Standard Mode (Real AI):**
```ini
PROJECT_ID="your_google_project_id"
DD_API_KEY="your_datadog_key"
DD_SITE="datadoghq.com"
DD_SERVICE="arbor-memory"
DD_ENV="hackathon"
```

*   `PROJECT_ID`: Your Google Cloud Project ID with Vertex AI enabled.
*   `DD_API_KEY`: Your Datadog API Key.

**For Mock Mode (Testing without keys):**
If you leave the `PROJECT_ID` or `DD_API_KEY` as placeholders (or empty), the app will automatically start in **Mock Mode**. This allows you to explore the UI and logic flow without needing real credentials.

## Running the Application

To start the Streamlit app, run:

```bash
streamlit run app.py
```

Navigate to the URL provided in the terminal (usually `http://localhost:8501`) to interact with Arbor.

### Interacting with Arbor

1.  **Start Chatting:** Begin a conversation in the "Start" node.
2.  **Context Drift:** Try changing the topic drastically (e.g., "I want to cook pasta").
3.  **Watch the Tree:**
    *   Arbor will detect the topic change and spawn a new branch.
    *   The sidebar visualization will update to show your **Active Path** (Root → Parent → Child) in **Green**.
    *   Inactive sibling branches will fade to **Grey**, indicating they are no longer polluting the AI's context.
