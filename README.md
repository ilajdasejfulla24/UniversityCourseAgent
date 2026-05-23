# University Course Selection Assistant

An agentic AI chatbot that helps students choose university courses based on interests, year of study, prerequisites, workload, career goals, and current opportunities.

This project satisfies the required stack:

- **LangGraph**: `StateGraph` with branching control flow.
- **LangChain**: LLM calls, prompt templates, tool definitions, and retriever integration.
- **RAG**: 17 internal university advising/course documents stored in Chroma.
- **Web search**: DuckDuckGo web search tool for fresh information.

## Architecture

```text
Student question
      |
      v
planner_node
      |
      |-- RAG only --> rag_node --> final_node
      |
      |-- Web only --> web_search_node --> final_node
      |
      |-- Both -----> rag_node --> web_search_node --> final_node
```

### Nodes

- `planner_node`: Uses an LLM with structured output to decide whether the question needs internal documents, web search, or both.
- `rag_node`: Retrieves relevant course/advising documents from Chroma and summarizes internal evidence.
- `web_search_node`: Searches the web for current information such as availability, scholarships, policies, internships, or job trends.
- `final_node`: Combines the evidence into a student-facing recommendation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```bash
GITHUB_TOKEN=your_github_models_token_here
GITHUB_MODELS_BASE_URL=https://models.github.ai/inference
GITHUB_MODEL=openai/gpt-4o-mini
```

Optional: set `TAVILY_API_KEY` if you later replace DuckDuckGo with Tavily.

## Build the vector store

```bash
python ingest.py
```

## Run the chatbot

```bash
python main.py
```

## Example query

```text
I am a second-year computer science student interested in AI. I have completed Python and Data Structures. What courses should I take next semester, and are there any current AI-related opportunities or scholarships?
```

Expected behavior:

1. The planner selects **both** RAG and web search.
2. RAG checks internal course descriptions, prerequisites, workload, and AI pathway documents.
3. Web search checks fresh AI opportunities or scholarships.
4. The final answer labels which information came from internal documents and which came from web search.

## Notes for demo

Good demo queries:

- RAG only: `I finished Python and Data Structures. Which AI courses can I take next?`
- Web only: `What are current AI scholarships for undergraduate students?`
- Both: `I am a second-year CS student interested in AI. I finished Python and Data Structures. What should I take next semester and are there current AI opportunities?`

## Academic integrity

See `AI_USAGE.md` for how AI assistance was used.

## Demo Video

https://drive.google.com/file/d/1d4JUsDUie0Y9I_JElqv29M7qmM_EMNlt/view?usp=drivesdk

