# Memory-Based AI Chatbot

A Streamlit-based chatbot with memory architecture featuring persona, short-term and long-term memory using FAISS vector search.

## 🏗️ Project Structure

```
New folder/
├── app.py                      # Main Streamlit application
├── .env                        # Environment variables (API keys)
├── requirements.txt            # Python dependencies
├── test_imports.py            # Test script to verify setup
├── vectorstore.py             # FAISS vector store implementation
├── config/
│   └── constants.py           # Configuration constants
├── core/
│   ├── api_client.py          # OpenRouter API client (commented out)
│   ├── context_assembler.py   # Prompt building with memory retrieval
│   └── fact_extractor.py      # Fact extraction and storage
├── data/                      # Data storage directory
│   ├── facts.json            # Stored facts
│   ├── memory.jsonl          # Short-term conversation memory
│   └── long_term_memory.jsonl # Long-term session summaries
├── memory/
│   ├── session_summarizer.py  # Session summarization
│   ├── long_term_memory.py    # Long-term memory management
│   ├── context_retriever.py   # FAISS-based memory retrieval
│   ├── fact_memory.py         # Fact memory utilities
│   └── turn_memory.py         # Turn-by-turn memory storage
├── persona/
│   ├── mood_adjustments.json  # Current mood settings
│   ├── relationship_status.py # User relationship tracking
│   ├── mood_tracker.py        # Mood tracking utilities
│   ├── persona.json           # AI persona definition
│   └── personality.json       # Personality traits
└── utils/
    ├── session_id.py          # Session management
    └── ui_helpers.py          # Streamlit UI utilities
```

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   - Copy `.env` and add your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

3. **Test the setup:**
   ```bash
   python test_imports.py
   ```

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 🧠 How It Works

### Memory Architecture
- **Short-term Memory**: Stores recent conversation turns in `memory.jsonl`
- **Long-term Memory**: Session summaries stored in `long_term_memory.jsonl`
- **FAISS Vector Search**: Retrieves relevant memories based on semantic similarity
- **Fact Storage**: Extracts and stores user facts in `facts.json`

### Persona System
- **Persona**: AI identity, goals, and pronouns
- **Personality**: Tone, style, temperament, formality
- **Mood**: Current emotional state with intensity tracking

### Prompt Assembly
1. Loads persona and personality traits
2. Retrieves relevant short-term memories via FAISS
3. Retrieves relevant long-term summaries
4. Combines everything into a contextual prompt

## 🔧 Current Status

- ✅ Memory storage and retrieval system
- ✅ FAISS vector search implementation  
- ✅ Persona and mood system
- ✅ Fact extraction and storage
- ✅ Session management
- 🚧 LLM integration (scaffolding in place, commented out)

## 🎯 Usage

1. Start chatting - your messages are stored and facts are extracted
2. The system builds contextual prompts using:
   - Your persona (Isabella)
   - Relevant conversation history
   - Your current mood and personality
3. Use "End Chat & Save to Long-Term Memory" to summarize sessions
4. View debug info in the sidebar

## 🔄 Next Steps

1. Uncomment LLM integration in `app.py` and `session_summarizer.py`
2. Add your OpenRouter API key to enable actual AI responses
3. Customize persona/personality files to match your preferences

## 🐛 Troubleshooting

- If imports fail, run `python test_imports.py` to diagnose
- Check that all `__init__.py` files are present
- Ensure the `data/` directory exists and is writable
- Verify your `.env` file is properly configured

## 📝 Notes

- Currently shows generated prompts instead of LLM responses
- FAISS embeddings use BGE-M3 model (downloaded on first run)
- All memory is stored locally in JSON/JSONL files
- Session IDs are generated per browser session
