# Arbor: Context Drift Visualization

Arbor is a Streamlit application that visualizes AI memory as a growing tree to prevent "context drift." It uses Google Vertex AI (Gemini Flash) for intelligence and Datadog LLM Observability for tracing and metrics.

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

Open the `.env` file and update the following values with your specific credentials:

```ini
PROJECT_ID="your_google_project_id"
DD_API_KEY="your_datadog_key"
DD_SITE="datadoghq.com"
DD_SERVICE="arbor-memory"
DD_ENV="hackathon"
```

*   `PROJECT_ID`: Your Google Cloud Project ID with Vertex AI enabled.
*   `DD_API_KEY`: Your Datadog API Key.
*   `DD_SITE`: Your Datadog site (e.g., `datadoghq.com`, `datadoghq.eu`).
*   `DD_SERVICE`: The service name for Datadog traces (default: `arbor-memory`).
*   `DD_ENV`: The environment tag for Datadog (default: `hackathon`).

## Running the Application

To start the Streamlit app, run:

```bash
streamlit run app.py
```

Navigate to the URL provided in the terminal (usually `http://localhost:8501`) to interact with Arbor.
