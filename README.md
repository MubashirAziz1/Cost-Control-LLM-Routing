# Cost Control Smart Model Router

A production-ready FastAPI service that intelligently routes user prompts to different LLM models based on complexity classification to optimize costs.

## Architecture

The service uses a two-stage approach:
1. **Classification**: A Phi-3 mini model classifies prompts as `simple`, `medium`, or `complex`
2. **Generation**: Based on classification, routes to:
   - **Simple** → Phi-3 mini (fast, cost-effective) 
   - **Medium** → Llama 3 70B (balanced performance)
   - **Complex** → GPT-OSS 120B (highest quality) 


## Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Groq API key 
- Ollama Application

### Installation

1. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   # Required
   GROQ_API_KEY=your_groq_api_key_here
   
   # Optional (defaults provided)
   CLASSIFIER_MODEL=microsoft/Phi-3-mini-4k-instruct
   SIMPLE_MODEL=microsoft/Phi-3-mini-4k-instruct
   MEDIUM_MODEL=meta-llama/Llama-3-70b-chat-hf
   COMPLEX_MODEL=gpt-oss-120b
   HOST=0.0.0.0
   PORT=8000
   LOG_LEVEL=INFO
   ```

## Running the Server

###  Using uvicorn directly
```bash
uvicorn api.main:app --reload
```

The server will start at `http://localhost:8000`

## API Usage

### Generate Endpoint

**POST** `/route`

**Request Body:**
```json
{
  "prompt": "What is the capital of France?"
}
```

**Response:**
```json
{
  "session_id": "rg098hjb873b",
  "sequence": 1,
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "model_name": "microsoft/Phi-3-mini-4k-instruct",
  "difficulty": "simple"
}
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`

## Design Decisions

1. **Modular Architecture**: Separate layers for API, routing, classification, and generation for easy maintenance and testing
2. **Lazy Loading**: Models are loaded only when needed to reduce memory usage
3. **PostGreSQL Database**: PostgreSql database is connected to maintain the logs and use it for conversational history
4. **Scalable Design**: Code structure allows easy addition of database and memory layers in the future
5. **Type Safety**: Uses Pydantic for request/response validation



