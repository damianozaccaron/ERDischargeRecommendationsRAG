EMBEDDING_MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"
NER_MODEL = "Clinical-AI-Apollo/Medical-NER"

# Table detection during ingestion. True uses the pymupdf-layout ML page segmentation 
USE_LAYOUT = True

# PDF extraction runs one process per guideline. 0 = auto (from core count and free RAM), 1 = serial.
INGESTION_WORKERS = 0

# True skips LLM generation, used to test retrieval quality.
RETRIEVAL_ONLY = False

# Use DeepL pre-translated files to save characters usage and ensure reproducibility.
USE_EXISTING_TRANSLATION = False

# Query building. False uses rule-based queries, true LLM-made ones. QUERY_MODEL picks the writer through OpenRouter
LLM_QUERIES = True
QUERY_MODEL = "anthropic/claude-opus-5.5"

# Translation of the note when USE_EXISTING_TRANSLATION is False.
LLM_TRANSLATION = not RETRIEVAL_ONLY or LLM_QUERIES

# Italian version of the recommendations and references (backend only). True has the output model translate them.
OUTPUT_LLM_TRANSLATION = True
# True appends the translation instructions to the generation prompt, False translates the finished output in a second call.
OUTPUT_TRANSLATION_IN_PROMPT = True


# Guideline scraper (code/scraping/): the model of every LLM step, with its own reasoning on.
GUIDELINES_MODEL = "anthropic/claude-opus-5.5"
GUIDELINES_BATCH = False

# Output LLM models
API = True
QUANT = True # if using local models, whether to use quantized versions from llama-cpp (if False, uses non-quantized models via transformers library)
REPO = "bartowski/Mistral-7B-Instruct-v0.3-GGUF"
MODEL_NAME = "anthropic/claude-opus-5.5"

# Retrieval and generation parameters
RETRIEVAL_K = 8    # per retrieval query
FINAL_N = 12
FAST_FINAL_N = 15    # excerpts without the cross-encoder (Streamlit fast mode)

TEMPERATURE = 0.2

PROMPT_MAX_CHARS = 32000

# Render the retrieved guideline tables (dosing, criteria, scores) in the home-discharge prompt.
USE_TABLES = True
TABLE_N = 5

# Tokens
OPENROUTER_KEY = ""
DEEPL_AUTH_KEY = ""
DEEPL_GLOSSARY_ID = "402df6de-7b21-464a-b3e7-114a03a0f8a4"