# Cost Control Smart Model Router

A production-ready FastAPI service that intelligently routes user prompts to different LLM models based on complexity classification to optimize costs.

## Architecture

The service uses a two-stage approach:
1. **Classification**: A Phi-3 mini model classifies prompts as `simple`, `medium`, or `complex`
2. **Generation**: Based on classification, routes to:
   - **Simple** → Phi-3 mini (fast, cost-effective)
   - **Medium** → Llama 3 70B (balanced performance)
   - **Complex** → GPT-4o (highest quality)

## Project Structure

```
Cost_Controller/
├── api/
│   └── main.py              # FastAPI application
├── router/
│   ├── classifier.py        # Prompt classification logic
│   ├── generator.py         # Model generation logic
│   └── router.py            # Routing orchestration
├── models/
│   └── schemas.py           # Pydantic schemas
├── config.py                # Configuration settings
├── main.py                  # Server entry point
├── pyproject.toml           # Dependencies (uv)
└── README.md
```

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- Hugging Face token (for accessing models)
- OpenAI API key (for GPT-4o)

### Installation

1. **Install uv** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e .
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```env
   # Required
   OPENAI_API_KEY=your_openai_api_key_here
   HF_TOKEN=your_huggingface_token_here
   
   # Optional (defaults provided)
   CLASSIFIER_MODEL=microsoft/Phi-3-mini-4k-instruct
   SIMPLE_MODEL=microsoft/Phi-3-mini-4k-instruct
   MEDIUM_MODEL=meta-llama/Llama-3-70b-chat-hf
   COMPLEX_MODEL=gpt-4o
   HOST=0.0.0.0
   PORT=8000
   LOG_LEVEL=INFO
   ```

## Running the Server

### Option 1: Using the main.py entry point
```bash
python main.py
```

### Option 2: Using uvicorn directly
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

## API Usage

### Generate Endpoint

**POST** `/generate`

**Request Body:**
```json
{
  "prompt": "What is the capital of France?"
}
```

**Response:**
```json
{
  "response": "The capital of France is Paris.",
  "model_name": "microsoft/Phi-3-mini-4k-instruct",
  "difficulty": "simple"
}
```

### Example with cURL

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing in simple terms"}'
```

### Example with Python

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={"prompt": "Write a Python function to calculate fibonacci numbers"}
)

data = response.json()
print(f"Model: {data['model_name']}")
print(f"Difficulty: {data['difficulty']}")
print(f"Response: {data['response']}")
```

## API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Design Decisions

1. **Modular Architecture**: Separate layers for API, routing, classification, and generation for easy maintenance and testing
2. **Lazy Loading**: Models are loaded only when needed to reduce memory usage
3. **Scalable Design**: Code structure allows easy addition of database and memory layers in the future
4. **Error Handling**: Comprehensive error handling with logging
5. **Type Safety**: Uses Pydantic for request/response validation

## Future Enhancements

The codebase is designed to be easily extended with:
- PostgreSQL database integration
- Memory/conversation history
- Caching layer
- Rate limiting
- Authentication/authorization
- Metrics and monitoring

## Notes

- Models are loaded lazily (on first use) to optimize startup time
- GPU support is automatic if CUDA is available
- The classifier model only classifies and does not generate the final response
- All models receive only the user prompt (no context or memory)
