"""
config.py
─────────────────────────────────────────────────────────────────────────────
Single source of truth for all tuneable parameters.

Covers four LLM providers (OpenAI, Anthropic, Google Gemini, Groq), the
Milvus vector store, PostgreSQL metadata store, and memory behaviour knobs
(similarity threshold, TTL, top-K retrieval).

Model catalogue
───────────────
``MODELS`` maps every short CLI label (e.g. ``"gpt-4.1-mini"``) to its
provider, the exact API model ID, and input/output token costs.  Model IDs
and prices were verified against each provider's live API in April 2026 and
were used for the benchmark reported in the accompanying paper.

Paper
─────
Subodh Kumar N. "Persistent Memory for Conversational AI Agents:
Architecture, Benchmarks, and Operational Thresholds."
GitHub: https://github.com/subodhkumar-n/memory-agent
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── OpenAI ────────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # text-embedding-3-large with MRL truncation: large-model quality at 1536-dim
    # Cost: ~$0.13/1M tokens vs $0.02 for small — negligible for research volume
    EMBED_MODEL: str = "text-embedding-3-large"   # used for ALL providers
    EMBED_DIM: int = 1536

    # ── Anthropic Claude ──────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # ── Google Gemini ─────────────────────────────────────────────────────
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

    # ── Groq (free) ───────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # ── All models available for experiments ──────────────────────────────
    # Model IDs verified live against each provider's API (April 2026).
    # Keys are short labels used in CLI and CSV output.
    MODELS: dict = {
        # ── OpenAI ──────────────────────────────────────────────────────
        "gpt-4o":          {"provider": "openai", "model_id": "gpt-4o",        "cost_per_1k_in": 0.0025,  "cost_per_1k_out": 0.010},
        "gpt-4o-mini":     {"provider": "openai", "model_id": "gpt-4o-mini",   "cost_per_1k_in": 0.00015, "cost_per_1k_out": 0.0006},
        "gpt-4.1":         {"provider": "openai", "model_id": "gpt-4.1",       "cost_per_1k_in": 0.002,   "cost_per_1k_out": 0.008},
        "gpt-4.1-mini":    {"provider": "openai", "model_id": "gpt-4.1-mini",  "cost_per_1k_in": 0.0004,  "cost_per_1k_out": 0.0016},
        "gpt-5":           {"provider": "openai", "model_id": "gpt-5",         "cost_per_1k_in": 0.010,   "cost_per_1k_out": 0.030},
        "gpt-5-mini":      {"provider": "openai", "model_id": "gpt-5-mini",    "cost_per_1k_in": 0.003,   "cost_per_1k_out": 0.010},
        "gpt-5.4":         {"provider": "openai", "model_id": "gpt-5.4",       "cost_per_1k_in": 0.015,   "cost_per_1k_out": 0.040},
        "gpt-5.4-pro":     {"provider": "openai", "model_id": "gpt-5.4-pro",   "cost_per_1k_in": 0.025,   "cost_per_1k_out": 0.060},
        # OpenAI reasoning (temperature fixed at 1)
        "o3":              {"provider": "openai", "model_id": "o3",            "cost_per_1k_in": 0.010,   "cost_per_1k_out": 0.040,  "temperature": 1},
        "o4-mini":         {"provider": "openai", "model_id": "o4-mini",       "cost_per_1k_in": 0.003,   "cost_per_1k_out": 0.012,  "temperature": 1},
        # ── Anthropic Claude ────────────────────────────────────────────
        "claude-haiku-4-5":  {"provider": "anthropic", "model_id": "claude-haiku-4-5-20251001",  "cost_per_1k_in": 0.0008, "cost_per_1k_out": 0.004},
        "claude-sonnet-4-5": {"provider": "anthropic", "model_id": "claude-sonnet-4-5-20250929", "cost_per_1k_in": 0.003,  "cost_per_1k_out": 0.015},
        "claude-sonnet-4-6": {"provider": "anthropic", "model_id": "claude-sonnet-4-6",          "cost_per_1k_in": 0.003,  "cost_per_1k_out": 0.015},
        "claude-opus-4-5":   {"provider": "anthropic", "model_id": "claude-opus-4-5-20251101",   "cost_per_1k_in": 0.015,  "cost_per_1k_out": 0.075},
        "claude-opus-4-6":   {"provider": "anthropic", "model_id": "claude-opus-4-6",            "cost_per_1k_in": 0.015,  "cost_per_1k_out": 0.075},
        # ── Google Gemini ─────────────────────────────────────────────────
        # All models below verified working on Google AI Studio Tier 1 (April 2026)
        # Gemini 2 family
        "gemini-2.5-flash-lite": {"provider": "gemini", "model_id": "gemini-2.5-flash-lite",        "cost_per_1k_in": 0.0001,  "cost_per_1k_out": 0.0004},
        "gemini-2.5-flash":      {"provider": "gemini", "model_id": "gemini-2.5-flash",             "cost_per_1k_in": 0.0003,  "cost_per_1k_out": 0.00125},
        "gemini-2.5-pro":        {"provider": "gemini", "model_id": "gemini-2.5-pro",               "cost_per_1k_in": 0.00125, "cost_per_1k_out": 0.010},
        # Gemini 3 family (Tier 1 required)
        "gemini-3-flash":        {"provider": "gemini", "model_id": "gemini-3-flash-preview",        "cost_per_1k_in": 0.0003,  "cost_per_1k_out": 0.0012},
        "gemini-3-pro":          {"provider": "gemini", "model_id": "gemini-3-pro-preview",          "cost_per_1k_in": 0.002,   "cost_per_1k_out": 0.008},
        "gemini-3.1-flash-lite": {"provider": "gemini", "model_id": "gemini-3.1-flash-lite-preview", "cost_per_1k_in": 0.0001,  "cost_per_1k_out": 0.0004},
        "gemini-3.1-pro":        {"provider": "gemini", "model_id": "gemini-3.1-pro-preview",        "cost_per_1k_in": 0.002,   "cost_per_1k_out": 0.010},
        # ── Groq — free tier (requires GROQ_API_KEY) ─────────────────────
        "llama3-70b":   {"provider": "groq", "model_id": "llama3-70b-8192",    "cost_per_1k_in": 0.0, "cost_per_1k_out": 0.0},
        "mixtral-8x7b": {"provider": "groq", "model_id": "mixtral-8x7b-32768", "cost_per_1k_in": 0.0, "cost_per_1k_out": 0.0},
    }

    @property
    def ACTIVE_MODELS(self) -> dict:
        """
        Subset of MODELS whose provider API key is available.
        Groq (and any other provider) is silently excluded when its key is absent.
        """
        key_map = {
            "openai":    self.OPENAI_API_KEY,
            "anthropic": self.ANTHROPIC_API_KEY,
            "gemini":    self.GOOGLE_API_KEY,
            "groq":      self.GROQ_API_KEY,
        }
        return {
            k: v for k, v in self.MODELS.items()
            if key_map.get(v["provider"], "")
        }

    # Default model used in the interactive chat
    DEFAULT_MODEL: str = "gpt-4o-mini"

    # ── PostgreSQL ────────────────────────────────────────────────────────
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_DB: str   = os.getenv("POSTGRES_DB", "memory_agent")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "secret")

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── Milvus Vector DB ──────────────────────────────────────────────────
    MILVUS_HOST: str       = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT: int       = int(os.getenv("MILVUS_PORT", 19530))
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "agent_memory")

    # ── Memory Behaviour (Phase-2 experiment knobs) ───────────────────────
    MEMORY_SIMILARITY_THRESHOLD: float = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", 0.75))
    MAX_MEMORIES_PER_QUERY: int        = int(os.getenv("MAX_MEMORIES_PER_QUERY", 10))
    MEMORY_TTL_DAYS: int               = int(os.getenv("MEMORY_TTL_DAYS", 30))

    MEMORY_TYPES = {
        "FACTUAL":   {"ttl_days": 365, "weight": 1.0},
        "PREFERENCE":{"ttl_days": 90,  "weight": 0.85},
        "EPHEMERAL": {"ttl_days": 1,   "weight": 0.3},
    }


settings = Settings()
