"""
Orthos — Local LLM Edition
STT (Whisper / Vosk)  +  Ollama LLM  +  TTS (EdgeTTS / Kokoro / ElevenLabs)
All Gemini / Google-AI dependencies removed.
"""
# ── Silence verbose logs + block heavy unused backends ─────────────────────
import os as _os
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")   # TensorFlow C++ noise
_os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")   # oneDNN banner
_os.environ.setdefault("GRPC_VERBOSITY",         "ERROR")
# USE_TF=0 prevents transformers from importing TensorFlow (saves 4-8 s).
# We intentionally do NOT set USE_TORCH or USE_JAX — forcing those values
# breaks transformers' lazy-loader on some versions (AutoModel disappears
# from the namespace).  Let transformers auto-detect the available backends.
_os.environ.setdefault("USE_TF",                 "0")
_os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Offline mode — use cached models, no HuggingFace network calls on startup.
# On first run the model isn't cached yet; tts.py / stt.py detect this and
# temporarily clear these flags to allow the one-time download, then they
# stay in effect for every subsequent launch (fully offline).
# ── Migrate from deprecated TRANSFORMERS_CACHE to HF_HOME ──────────────────
_hf_cache = _os.environ.pop("TRANSFORMERS_CACHE", None)
if _hf_cache:
    _os.environ.setdefault("HF_HOME", _hf_cache)
_os.environ.setdefault("HF_HOME", _os.path.join(_os.environ.get("USERPROFILE", "C:\\Users\\default"), ".cache\\huggingface"))
# ───────────────────────────────────────────────────────────────────────────
_os.environ.setdefault("HF_HUB_OFFLINE",      "1")
_os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
_os.environ.setdefault("HF_DATASETS_OFFLINE",  "1")
import warnings as _warnings
_warnings.filterwarnings("ignore", category=UserWarning)
_warnings.filterwarnings("ignore", category=DeprecationWarning)
_warnings.filterwarnings("ignore", category=FutureWarning)
# ───────────────────────────────────────────────────────────────────────────

# ── Windows console safety ─────────────────────────────────────────────────
# Printing unicode (→, ⚠️, emoji) raises 'charmap' codec errors when
# stdout/stderr is piped or redirected (which would crash the Gemini Live
# session mid-tool-round).  Keep the detected encoding, never raise.
import sys as _sys
for _stream in (_sys.stdout, _sys.stderr):
    try:
        if _stream is not None and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(errors="replace")
    except Exception:
        pass

# ── Global exception handler (catch ALL unhandled exceptions) ───────────────
import traceback as _traceback
import threading as _threading

def _global_excepthook(exc_type, exc_value, exc_tb):
    """Print FULL traceback for any unhandled exception."""
    print("=" * 70, file=sys.stderr)
    print("UNHANDLED EXCEPTION (main thread):", file=sys.stderr)
    _traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    print("=" * 70, file=sys.stderr)

def _thread_excepthook(args):
    """Print FULL traceback for unhandled exceptions in threads."""
    print("=" * 70, file=sys.stderr)
    print(f"UNHANDLED EXCEPTION in thread '{args.thread.name}':", file=sys.stderr)
    _traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=sys.stderr)
    print("=" * 70, file=sys.stderr)

import sys
sys.excepthook = _global_excepthook
_threading.excepthook = _thread_excepthook
# ───────────────────────────────────────────────────────────────────────────

# ── CLI arguments ─────────────────────────────────────────────────────────
import argparse as _argparse
from pathlib import Path as _Path

_CAPTURES_DIR = _Path(__file__).resolve().parent / "memory" / "captures"
_CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

_ARGS = _argparse.ArgumentParser(description="Orthos — AI Assistant")
_ARGS.add_argument("--headless", "--no-gui", action="store_true",
                    help="Run without PyQt6 GUI (Telegram-only terminal mode)")
_ARGS = _ARGS.parse_args()
# ───────────────────────────────────────────────────────────────────────────

# ── Bootstrap: auto-install base UI packages before anything else ──────────
# Uses only stdlib so it works even on a completely fresh Python install.
import importlib.util as _ilu
import subprocess      as _sp
import sys             as _sys

_BASE_PKGS = [
    ("psutil",      "psutil"),
    ("numpy",       "numpy"),
    ("PIL",         "pillow"),
    ("requests",    "requests"),
]
if not _ARGS.headless:
    _BASE_PKGS.extend([
        ("PyQt6",       "PyQt6"),
        ("sounddevice", "sounddevice"),
    ])

def _bootstrap() -> None:
    need = [pkg for mod, pkg in _BASE_PKGS if _ilu.find_spec(mod) is None]
    if not need:
        return
    print(f"\n[Orthos] First-run setup — installing: {', '.join(need)}")
    print("[Orthos] This happens only once.\n")
    _sp.run([_sys.executable, "-m", "pip", "install", *need], check=True)
    print("\n[Orthos] Base packages ready — restarting…\n")
    # Replace current process with a fresh one (picks up newly installed packages)
    _os.execv(_sys.executable, [_sys.executable] + _sys.argv)

_bootstrap()
# ───────────────────────────────────────────────────────────────────────────

import base64
import json
import io
import queue
import re
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

from ui import OrthosUI
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
    format_memory_items_from_sqlite,
    save_conversation, load_recent_conversations,
    create_session, list_sessions, get_session, rename_session,
    close_and_summarize, load_session_turns, load_all_summaries,
    auto_title_session, delete_session,
    set_session_memory, get_session_memory, format_session_memory,
    get_session_recent_turns,
    log_timeline_event, format_timeline, auto_detect_timeline_events,
    search_timeline,
    # ── Memory items CRUD (used by cleanup worker) ──
    get_memory_items,
    update_memory_item,
    # ── New retrieval-first ──
    hybrid_search,
    invalidate_search_cache,
    format_memory_items_for_prompt,
    add_memory_item,
    fork_session,
    compress_oldest_turns,
    estimate_session_tokens,
    # ── Project ──
    _ensure_default_hierarchy,
    list_projects,
    create_project,
    get_project,
    set_session_project,
    # ── Memory versioning ──
    save_memory_version,
    get_memory_versions,
    # ── Tool memory ──
    set_tool_memory,
    get_tool_memory,
    # ── Entity / Graph ──
    get_entity_graph,
    # ── Logging ──
    log_retrieval,
    log_prompt,
)
from core.llm_client import call_llm, call_llm_stream, get_llm_settings, _get_provider_config

try:
    from core.gemini_live_provider import QuotaExceededError as _QuotaExceededError
except Exception:
    _QuotaExceededError = None

from actions.file_processor    import file_processor
from actions.flight_finder     import flight_finder
from actions.open_app          import open_app
from actions.weather_report    import weather_action
from actions.send_message      import send_message
from actions.reminder          import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor  import screen_process, screen_locate, _capture_screen_annotated, _get_screenshot_quality
from actions.youtube_video     import youtube_video
from actions.desktop           import desktop_control
from actions.file_controller   import file_controller
from actions.code_helper       import code_helper
from actions.dev_agent         import dev_agent
from actions.web_search        import web_search as web_search_action, webfetch as webfetch_action
from actions.computer_control  import computer_control
from actions.game_updater      import game_updater
from actions.terminal          import run_terminal

# ── Improved modules ───────────────────────────────────────────────────
from core.logging_setup import setup_logging, get_logger, patch_print, catch_exceptions
from core.security import SecretsManager, Sanitizer
from core.tool_registry import discover as discover_tools
from core.llm_provider import get_default_provider
from core.observability import MetricsCollector, HealthCheck, start_metrics_server, Timer, init_tracing
from memory.chroma_memory import ChromaMemory, MemoryBridge, CHROMA_VECTOR_SEARCH_ENABLED
from core.mcp_manager import MCPManager

# ── Global logging ─────────────────────────────────────────────────────
logger = get_logger("mark")
setup_logging()
patch_print()  # replaces all print() calls with loguru

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


BASE_DIR        = _get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"

SAMPLE_RATE_IN = 16_000
BLOCK_SIZE     = 1_024
CHANNELS       = 1

# ---------------------------------------------------------------------------
# Secrets manager
# ---------------------------------------------------------------------------

_secrets = SecretsManager()


def _load_config() -> dict:
    try:
        cfg = json.loads(API_CONFIG_PATH.read_text(encoding="utf-8"))
        for key in list(cfg.keys()):
            env_val = _secrets.get(key.upper())
            if env_val:
                cfg[key] = env_val
        return cfg
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Dynamic tool registry
# ---------------------------------------------------------------------------

_tool_registry = discover_tools()
# If no dynamically discovered tools (no @tool decorators on action files),
# fall back to the static legacy TOOL_DECLARATIONS from main.py.original
if not _tool_registry.list_tools():
    from core.tool_declarations import TOOL_DECLARATIONS as _LEGACY_DECLARATIONS
    _tool_registry.register_legacy(_LEGACY_DECLARATIONS)
TOOL_DECLARATIONS = _tool_registry.get_declarations()
OLLAMA_TOOLS = TOOL_DECLARATIONS

# ── Dynamic Tool Index (semantic search + on-demand schema loading) ──────
from core.tool_index import ToolIndex, TOOL_SUMMARIES, META_TOOL_DEFINITIONS, \
    make_search_tools_handler, make_get_tool_definition_handler

_tool_index = ToolIndex(_tool_registry)
_tool_active_full_schemas: set[str] = set()

# Register meta-tools so they can be executed
_search_tools_fn = make_search_tools_handler(_tool_index)
_get_tool_def_fn = make_get_tool_definition_handler(_tool_index, _tool_active_full_schemas)
_tool_registry.register_tool("search_tools", _search_tools_fn,
    description="Search for available tools by capability. Returns matching tool names, descriptions, and full parameter schemas.",
    parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING", "description": "What capability you need"}, "k": {"type": "INTEGER", "description": "Number of results (default 5, max 10)"}}, "required": ["query"]})
_tool_registry.register_tool("get_tool_definition", _get_tool_def_fn,
    description="Load the full parameter schema for a specific tool. Call this when search_tools shows a matching tool and you need its complete parameters.",
    parameters={"type": "OBJECT", "properties": {"tool_name": {"type": "STRING", "description": "Exact tool name e.g. 'web_search'"}}, "required": ["tool_name"]})


def _build_dynamic_tools() -> list[dict]:
    """Build the tools payload: meta-tools + internal tools (full schemas) + loaded MCP tools.
    
    Internal tools: always sent with full parameter schemas (not empty {}).
    MCP tools: NEVER sent in payload. Only available via search_tools, loaded on demand.
    """
    tools = list(META_TOOL_DEFINITIONS)

    # Internal tools — always send full schemas
    for name in _tool_index._tool_names:
        full = _tool_index.get_tool_definition(name)
        if full:
            tools.append(full)
        else:
            tools.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": TOOL_SUMMARIES.get(name, name),
                    "parameters": {"type": "object", "properties": {}},
                }
            })

    # MCP tools — only if LLM explicitly loaded via get_tool_definition
    for name in _tool_index._mcp_tool_names:
        if name in _tool_active_full_schemas:
            full = _tool_index.get_tool_definition(name)
            if full:
                tools.append(full)

    return tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are Orthos, an AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )


# ---------------------------------------------------------------------------
# Voice Activity Detection (used for Whisper listen loop)
# ---------------------------------------------------------------------------

class _VADBuffer:
    """Adaptive energy-based VAD: buffers audio until end of utterance.

    Continuously tracks the noise floor so it works with any mic
    (including WO Mic which has a high constant noise floor ~0.24 RMS).
    Speech is detected when RMS significantly exceeds the noise floor.
    """

    def __init__(
        self,
        sample_rate:    int   = 16_000,
        silence_sec:    float = 3.0,     # silence after last word → send to STT
        min_speech_sec: float = 0.3,
        max_speech_sec: float = 30.0,
    ):
        self._sr        = sample_rate
        self._sil_n     = int(silence_sec * sample_rate)
        self._min_n     = int(min_speech_sec * sample_rate)
        self._max_n     = int(max_speech_sec * sample_rate)
        self._buf: list[np.ndarray] = []
        self._in_spch   = False
        self._sil_cnt   = 0
        self._history: list[float] = []
        self._noise_floor = 0.0

    def _update_noise_floor(self, rms: float) -> tuple[float, float]:
        self._history.append(rms)
        if len(self._history) > 30:
            self._history.pop(0)
        self._noise_floor = float(np.median(self._history))
        speech_thresh = self._noise_floor * 1.5
        silence_thresh = self._noise_floor * 1.15
        return speech_thresh, silence_thresh

    def process(self, chunk: np.ndarray) -> np.ndarray | None:
        """
        Feed one audio chunk (float32 mono).
        Returns complete utterance when speech ends, otherwise None.

        Thresholds adapt to the current noise floor:
          - speech starts when RMS > noise_floor × 1.5
          - speech ends when RMS < noise_floor × 1.15
        The gap between the two prevents mid-sentence cuts.
        """
        rms     = float(np.sqrt(np.mean(chunk ** 2)))
        speech_thresh, silence_thresh = self._update_noise_floor(rms)
        total_n = sum(len(c) for c in self._buf)

        if rms > speech_thresh:
            self._in_spch = True
            self._sil_cnt = 0
            self._buf.append(chunk.copy())
            if total_n >= self._max_n:
                audio         = np.concatenate(self._buf)
                self._buf     = []
                self._in_spch = False
                self._sil_cnt = 0
                if len(audio) >= self._min_n:
                    return audio
        elif self._in_spch:
            self._buf.append(chunk.copy())
            if rms < silence_thresh:
                self._sil_cnt += len(chunk)

            if self._sil_cnt >= self._sil_n or total_n >= self._max_n:
                audio         = np.concatenate(self._buf)
                self._buf     = []
                self._in_spch = False
                self._sil_cnt = 0
                if len(audio) >= self._min_n:
                    return audio
        return None


class _BargeDetector:
    """Detects real speech while Orthos is speaking (barge-in).

    Only audio well above the ambient noise floor triggers an interrupt,
    so speaker echo of Orthos's own voice does not self-trigger. The noise
    floor adapts via a rolling median over the last ~3 seconds.
    """

    def __init__(self, floor_rms: float = 0.02, history: int = 30):
        self._floor_rms = float(floor_rms)
        self._hist: list[float] = []

    def speech(self, rms: float) -> bool:
        self._hist.append(rms)
        if len(self._hist) > 30:
            self._hist.pop(0)
        noise = max(self._floor_rms, float(np.median(self._hist)) * 2.5)
        return rms > noise


# ---------------------------------------------------------------------------
# Conversation trim policy (2026-09-06, user-decided numbers)
# ---------------------------------------------------------------------------
# Raw conversation hits 50k tokens → summarize ONLY the oldest ~15k into the
# rolling summary; the newest ~35k stay raw in the prompt so recent exchanges
# keep full fidelity (the old behaviour summarized the ENTIRE conversation,
# which caused "context amnesia" after every trim).
from core.conversation_trim import (
    RAW_CONV_TRIGGER_TOKENS as _RAW_CONV_TRIGGER_TOKENS,
    TRIM_CHUNK_TOKENS as _TRIM_CHUNK_TOKENS,
    oldest_trim_indices as _oldest_trim_split,
)


# ---------------------------------------------------------------------------
# Orthos
# ---------------------------------------------------------------------------

class Orthos:
    """
    Main assistant class.
    Replaces JarvisLive (Gemini Live API) with:
      STT (Whisper/Vosk) → Ollama LLM (tool calling) → TTS (Edge/Kokoro/ElevenLabs)
    """

    def __init__(self, ui: OrthosUI):
        self.ui               = ui
        self._config          = _load_config()
        self._stt             = None
        self._tts             = None
        self._tts_ready       = threading.Event()
        self._speaking        = False
        self._speaking_lock   = threading.Lock()
        self._should_stop     = threading.Event()
        self._cancel_event    = threading.Event()
        self._cancel_seq      = 0
        self._cancel_lock     = threading.Lock()
        self._text_queue:     queue.Queue = queue.Queue()
        self._tts_queue:      queue.Queue = queue.Queue()
        self._audio_queue:    queue.Queue = queue.Queue(maxsize=2)
        self._streaming_tts   = False       # True when TTS engine yields PCM chunks (Gemini)
        self._out_stream      = None        # sd.OutputStream for streaming playback
        self._conversation:   list[dict]  = []
        self._conv_saved_idx  = 0
        self._rolling_summary: str = ""
        self._conv_budget: int = 5000
        self._summary_rolling_budget: int = 50000
        self._summary_past_budget: int = 100000
        self._cur_session_id: str | None = None
        self._cur_project_id: str | None = None
        self._session_list: list[dict] = []
        self._response_callbacks: list = []
        self._gateway = None
        self._is_headless = _ARGS.headless

        # ── 5 Background worker queues ──
        self._embedding_queue: queue.Queue = queue.Queue()
        self._summary_queue: queue.Queue = queue.Queue()
        self._reflection_queue: queue.Queue = queue.Queue()
        self._extraction_queue: queue.Queue = queue.Queue()
        self._cleanup_queue: queue.Queue = queue.Queue()
        self._start_background_workers()

        # ── Ensure default project hierarchy exists ──
        try:
            _ensure_default_hierarchy()
        except Exception:
            pass

        # ── Improved module integrations ──────────────────────────────────
        self._metrics = MetricsCollector()
        self._health = HealthCheck()
        self._chroma = ChromaMemory() if CHROMA_VECTOR_SEARCH_ENABLED else None
        self._memory_bridge = MemoryBridge()
        self._provider = get_default_provider()
        self._sanitizer = Sanitizer()
        self._secrets = _secrets

        self.ui.on_text_command = self._on_text_command
        self.ui.on_text_command_with_files = self._on_text_command_with_files
        # Connect session UI callbacks
        try:
            self.ui._win._session_cb_new = self._new_session
            self.ui._win._session_cb_switch = self._switch_session
            self.ui._win._session_cb_delete = self._delete_session
            self.ui._win._session_cb_rename = self._rename_session_ui
            self.ui._win._session_cb_fork = self._fork_session_ui
            # ── Project & memory version callbacks ──
            self.ui._win._project_cb_list = self._list_projects_ui
            self.ui._win._project_cb_switch = self._switch_project_ui
            self.ui._win._project_cb_create = self._create_project_ui
            self.ui._win._version_cb_browse = self._browse_memory_versions_ui
            # ── Inline chat actions (F7): Regenerate / Continue ──
            if hasattr(self.ui._win, "_log"):
                self.ui._win._log._on_action_cb = self._handle_chat_action
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Response delivery (UI + Telegram gateway fan-out)
    # ------------------------------------------------------------------

    def register_response_callback(self, callback) -> None:
        self._response_callbacks.append(callback)

    def _deliver_response(self, text: str, source: str = "ui", quiet: bool = False) -> None:
        if not text:
            return
        if not quiet:
            self.ui.write_log(f"Orthos: {text}")
        for cb in self._response_callbacks:
            try:
                cb(text, source)
            except Exception as e:
                print(f"[ResponseRouter] callback error: {e}")

    # ------------------------------------------------------------------
    # Async reflection queue
    # ------------------------------------------------------------------

    def _start_background_workers(self) -> None:
        """Start 5 background worker threads: embedding, summary, reflection, extraction, cleanup."""

        def _embedding_worker():
            while True:
                try:
                    task = self._embedding_queue.get(timeout=60)
                except queue.Empty:
                    continue
                if task is None:
                    break
                try:
                    item_id = task.get("item_id", 0)
                    content = task.get("content", "")
                    if item_id and content:
                        from memory.chroma_memory import ChromaMemory, CHROMA_VECTOR_SEARCH_ENABLED
                        if CHROMA_VECTOR_SEARCH_ENABLED:
                            chroma = ChromaMemory()
                            chroma.add_memory("memory_items", str(item_id), content)
                except Exception as e:
                    print(f"[Worker] Embedding error: {e}")

        def _summary_worker():
            while True:
                try:
                    task = self._summary_queue.get(timeout=60)
                except queue.Empty:
                    continue
                if task is None:
                    break
                try:
                    session_id = task.get("session_id", self._cur_session_id)
                    compress_oldest_turns(session_id)
                except Exception as e:
                    print(f"[Worker] Summary error: {e}")

        def _reflection_worker():
            while True:
                try:
                    task = self._reflection_queue.get(timeout=60)
                except queue.Empty:
                    continue
                if task is None:
                    break
                try:
                    session_id = task.get("session_id", self._cur_session_id)
                    text = task.get("text", "")
                    if session_id and text:
                        log_timeline_event(
                            session_id, "reflection",
                            f"Async reflection: {text[:200]}",
                            auto=True,
                        )
                except Exception as e:
                    print(f"[Worker] Reflection error: {e}")

        def _extraction_worker():
            while True:
                try:
                    task = self._extraction_queue.get(timeout=60)
                except queue.Empty:
                    continue
                if task is None:
                    break
                try:
                    text = task.get("text", "")
                    if text:
                        self._extract_session_memory(text)
                except Exception as e:
                    print(f"[Worker] Extraction error: {e}")

        def _cleanup_worker():
            """Aging: decay scores for memory_items not recently accessed."""
            while True:
                try:
                    task = self._cleanup_queue.get(timeout=3600)  # run hourly
                except queue.Empty:
                    pass
                if task is None:
                    break
                try:
                    items = get_memory_items(limit=200)
                    now = datetime.now()
                    for item in items:
                        last_acc = item.get("last_access", "")
                        if last_acc:
                            days = (now - datetime.fromisoformat(last_acc)).days
                        else:
                            days = 365
                        if days > 90:
                            new_decay = max(0.1, item.get("decay_score", 1.0) - 0.1 * (days / 30))
                            update_memory_item(item["id"], importance=new_decay * item.get("importance", 1.0))
                except Exception as e:
                    print(f"[Worker] Cleanup error: {e}")

        threads = [
            ("embedding", _embedding_worker),
            ("summary", _summary_worker),
            ("reflection", _reflection_worker),
            ("extraction", _extraction_worker),
            ("cleanup", _cleanup_worker),
        ]
        for name, target in threads:
            t = threading.Thread(target=target, daemon=True)
            t.start()

    def _enqueue_reflection(self, text: str, session_id: str | None = None) -> None:
        """Queue a reflection task for async processing."""
        try:
            self._reflection_queue.put_nowait({
                "text": text,
                "session_id": session_id or self._cur_session_id,
            })
        except Exception:
            pass

    def _enqueue_embedding(self, item_id: int, content: str) -> None:
        try:
            self._embedding_queue.put_nowait({"item_id": item_id, "content": content})
        except Exception:
            pass

    def _enqueue_extraction(self, text: str) -> None:
        try:
            self._extraction_queue.put_nowait({"text": text})
        except Exception:
            pass

    def _enqueue_summary(self, session_id: str | None = None) -> None:
        try:
            self._summary_queue.put_nowait({"session_id": session_id or self._cur_session_id})
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _init_session_system(self) -> None:
        """Called on startup. Creates a fresh session, loads past summaries."""
        self._cur_project_id = _ensure_default_hierarchy()[-1]  # get default project id
        # Migrate old sessions (project_id IS NULL) to default project
        try:
            import sqlite3
            from memory.conversation_db import DB_PATH
            with sqlite3.connect(str(DB_PATH)) as _conn:
                _conn.execute(
                    "UPDATE sessions SET project_id = ? WHERE project_id IS NULL",
                    (self._cur_project_id,),
                )
                _conn.commit()
        except Exception:
            pass
        self._session_list = list_sessions(project_id=self._cur_project_id)
        self._cur_session_id = create_session("New Chat", project_id=self._cur_project_id)
        self._conversation = []
        self._conv_saved_idx = 0
        self._rolling_summary = ""
        try:
            self.ui._win._cur_session_id = self._cur_session_id
            self.ui._win._cur_project_id = self._cur_project_id
            self.ui._win._session_refresh_sig.emit()
        except Exception:
            pass
        print(f"[Session] Started: {self._cur_session_id}")

    def _new_session(self) -> None:
        """Close current session and start a new one."""
        if self._cur_session_id:
            self._close_current_session()
        self._cur_session_id = create_session("New Chat", project_id=self._cur_project_id)
        self._conversation = []
        self._conv_saved_idx = 0
        self._rolling_summary = ""
        self._session_list = list_sessions(project_id=self._cur_project_id)
        try:
            self.ui._win._cur_session_id = self._cur_session_id
            self.ui._win._refresh_session_list()
            self.ui._win._clear_log()
        except Exception:
            pass
        print(f"[Session] New session started: {self._cur_session_id}")

    def _switch_session(self, session_id: str) -> None:
        """Switch to an existing session. Close current, load target."""
        if session_id == self._cur_session_id:
            return
        if self._cur_session_id:
            self._close_current_session()
        self._cur_session_id = session_id
        self._conversation = load_session_turns(session_id)
        self._conv_saved_idx = len(self._conversation)
        try:
            sm = get_session_memory(session_id, "_rolling_summary")
            self._rolling_summary = sm.get("_rolling_summary", "") if sm else ""
        except Exception:
            self._rolling_summary = ""
        # ── Trim conversation when rolling summary exists (avoid 2x payload) ──
        if self._rolling_summary and len(self._conversation) > 3:
            try:
                from memory.conversation_db import _estimate_tokens
                total = 0
                trimmed = []
                for t in reversed(self._conversation):
                    t_tok = _estimate_tokens(t.get("content", "") or "")
                    if total + t_tok > 10000:
                        break
                    trimmed.insert(0, t)
                    total += t_tok
                if trimmed:
                    self._conversation = trimmed
                    self._conv_saved_idx = len(trimmed)
            except Exception:
                pass
        self._session_list = list_sessions(project_id=self._cur_project_id)
        try:
            self.ui._win._cur_session_id = self._cur_session_id
            self.ui._win._refresh_session_list()
            self.ui._win._populate_log(self._conversation)
        except Exception as e:
            print(f"[Session] Failed to restore chat log: {e}")
            traceback.print_exc()
        session = get_session(session_id)
        title = session["title"] if session else session_id
        print(f"[Session] Switched to: {title} ({session_id})")

    def _close_current_session(self) -> None:
        """Save remaining turns, close and summarize the current session."""
        if not self._cur_session_id:
            return
        try:
            new_turns = self._conversation[self._conv_saved_idx:]
            if new_turns:
                save_conversation(new_turns, self._cur_session_id)
            summary = close_and_summarize(self._cur_session_id)
            if summary:
                print(f"[Session] Summary: {summary[:80]}...")
            # Auto-title from first user message if still "New Chat"
            session = get_session(self._cur_session_id)
            if session and session.get("title") == "New Chat":
                auto_title_session(self._cur_session_id)
        except Exception as e:
            print(f"[Session] Close error: {e}")
        self._cur_session_id = None

    def _delete_session(self, session_id: str) -> None:
        """Delete a session and all its data."""
        if session_id == self._cur_session_id:
            self._close_current_session()
            delete_session(session_id)
            self._cur_session_id = create_session("New Chat", project_id=self._cur_project_id)
            self._conversation = []
            self._conv_saved_idx = 0
        else:
            delete_session(session_id)
        self._session_list = list_sessions(project_id=self._cur_project_id)
        try:
            self.ui._win._refresh_session_list()
        except Exception:
            pass

    def _rename_session(self, session_id: str, title: str) -> None:
        rename_session(session_id, title)
        self._session_list = list_sessions(project_id=self._cur_project_id)
        try:
            self.ui._win._refresh_session_list()
        except Exception:
            pass

    def _rename_session_ui(self, session_id: str) -> None:
        """Show rename dialog and apply."""
        try:
            from PyQt6.QtWidgets import QInputDialog
            session = get_session(session_id)
            old_title = session["title"] if session else "Chat"
            new_title, ok = QInputDialog.getText(
                None, "Rename Session", "New title:",
                text=old_title,
            )
            if ok and new_title and new_title.strip():
                self._rename_session(session_id, new_title.strip())
        except Exception as e:
            print(f"[Session] Rename dialog error: {e}")

    def _fork_session_ui(self, session_id: str) -> None:
        """Fork a session: create new session from current state of the fork source."""
        try:
            from memory.conversation_db import fork_session
            new_id = fork_session(session_id, from_turn_id=0, title=f"Fork of {session_id[:8]}")
            # Switch to the new fork
            self._switch_session(new_id)
            print(f"[Session] Forked {session_id} → {new_id}")
        except Exception as e:
            print(f"[Session] Fork error: {e}")

    # ── Project management ─────────────────────────────────────────────

    def _list_projects_ui(self) -> list[dict]:
        return list_projects()

    def _switch_project_ui(self, project_id: str) -> None:
        """Switch current project view — reload session list for the new project."""
        self._cur_project_id = project_id
        self._session_list = list_sessions(project_id=project_id)
        # If current session belongs to old project, switch to most recent in new project
        if self._session_list:
            self._switch_session(self._session_list[0]["id"])
        else:
            if self._cur_session_id:
                self._close_current_session()
            self._cur_session_id = create_session("New Chat", project_id=project_id)
            self._conversation = []
            self._conv_saved_idx = 0
        try:
            self.ui._win._cur_project_id = project_id
            self.ui._win._refresh_session_list()
            if not self._session_list:
                self.ui._win._clear_log()
        except Exception:
            pass
        proj = get_project(project_id)
        pname = proj["name"] if proj else project_id
        print(f"[Project] Switched to: {pname} ({project_id})")

    def _create_project_ui(self, name: str) -> str:
        pid = create_project(name)
        self._session_list = list_sessions(project_id=self._cur_project_id)
        try:
            self.ui._win._refresh_session_list()
        except Exception:
            pass
        return pid

    def _browse_memory_versions_ui(self) -> list[dict]:
        return get_memory_versions()

    # ── Session memory extraction ─────────────────────────────────────────

    def _extract_session_memory(self, text: str) -> None:
        """Parse AI output for goals, issues, code, files, todos and store in session_memory."""
        if not self._cur_session_id:
            return
        import re
        patterns = {
            "goal":     r"(?:current\s+)?(?:goal|objective|aim)[:\s]+(.+?)(?=\n\n|\n(?:\w+\s*:)|\Z)",
            "issue":    r"(?:issue|blocker|problem|challenge)[:\s]+(.+?)(?=\n\n|\n(?:\w+\s*:)|\Z)",
            "code":     r"(?:code|function|class|script)[:\s]+(.+?)(?=\n\n|\n(?:\w+\s*:)|\Z)",
            "files":    r"(?:files?|file\s+created|edited|modified)[:\s]+(.+?)(?=\n\n|\n(?:\w+\s*:)|\Z)",
            "todo":     r"(?:todo|next\s+step|remaining)[:\s]+(.+?)(?=\n\n|\n(?:\w+\s*:)|\Z)",
        }
        for key, pat in patterns.items():
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                val = m.group(1).strip()[:500]
                if val:
                    set_session_memory(self._cur_session_id, key, val)

    def _extract_and_log(self, text: str) -> None:
        """Call after each assistant response to update session memory + timeline + entity graph."""
        self._extract_session_memory(text)
        if self._cur_session_id:
            try:
                _store_count = getattr(self, "_memory_extract_count", 0)
                if _store_count % 5 == 0:  # every 5 turns, detect timeline events
                    from memory.conversation_db import get_session_recent_turns as _gsrt
                    recent = _gsrt(self._cur_session_id)
                    auto_detect_timeline_events(self._cur_session_id, recent,
                                                project_id=self._cur_project_id or "")
                setattr(self, "_memory_extract_count", _store_count + 1)
            except Exception:
                pass
        # Extract entities for knowledge graph (async via queue).
        # Dedup by content hash — the same response used to be linked
        # dozens of times, flooding the graph with boilerplate edges.
        try:
            import hashlib as _hashlib
            _h = _hashlib.md5((text or "").encode("utf-8")).hexdigest()[:12]
            _recent = getattr(self, "_kg_resp_hashes", [])
            if _h not in _recent and (text or "").strip():
                from memory.entity_store import link_fact
                link_fact("conversation", "recent_response", text[:500])
                _recent = (_recent + [_h])[-3:]
                setattr(self, "_kg_resp_hashes", _recent)
        except Exception:
            pass

    # ── Reflection & Decay ───────────────────────────────────────────────

    def _build_reflection(self) -> str:
        """Build a short reflection summary from recent conversation turns."""
        try:
            recent = self._conversation[-6:]
            lines = []
            for msg in recent:
                role = msg.get("role", "?")
                content = (msg.get("content") or "")[:120]
                lines.append(f"[{role}] {content}")
            return "; ".join(lines[-3:]) or "No recent content"
        except Exception:
            return "Reflection unavailable"

    def _decay_old_memories(self) -> None:
        """Decay importance of session memory items not recently updated."""
        try:
            from memory.conversation_db import get_session_recent_turns as _gsrt
            recent = _gsrt(self._cur_session_id, limit=3)
            recent_text = " ".join(t.get("content", "") for t in recent).lower()
            if self._cur_session_id:
                from memory.conversation_db import get_session_memory as _gsm
                mem = _gsm(self._cur_session_id)
                for key, value in mem.items():
                    # NEVER decay internal framework keys — `_rolling_summary`
                    # and `_summary_*` carry the persisted older-context and
                    # deleting them silent-wipes the model's history memory.
                    if key.startswith("_"):
                        continue
                    if key not in recent_text and len(self._conversation) > 20:
                        log_timeline_event(
                            self._cur_session_id, "decay",
                            f"Session memory '{key}' faded (not recently referenced)",
                            auto=True,
                        )
                        from memory.conversation_db import delete_session_memory as _dsm
                        _dsm(self._cur_session_id, key)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        sys_p       = _load_system_prompt()
        now         = datetime.now()
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {now.strftime('%A, %B %d, %Y — %I:%M %p')}\n"
            f"Use this to calculate exact times for reminders."
        )

        parts = [sys_p]

        # Extract query from last user message for contextual retrieval
        query = ""
        for msg in reversed(self._conversation):
            if msg.get("role") == "user":
                query = (msg.get("content") or "")[:200]
                break

        # 1. Past session summaries (raw turns are in messages array — no duplicate)
        if self._cur_session_id:
            try:
                from memory.conversation_db import get_session_context
                conv_ctx = get_session_context(self._cur_session_id, include_summaries=True, include_raw_turns=False, max_tokens=self._summary_past_budget)
                if conv_ctx:
                    parts.append(conv_ctx)
            except Exception:
                pass

        # 1b. Current session rolling summary (oldest turns summarized to stay within budget)
        if self._rolling_summary:
            parts.append(f"[CURRENT SESSION SUMMARY — older context summarized]\n{self._rolling_summary}")

        # 2. Session memory (current goals, issues, code, files)
        if self._cur_session_id:
            try:
                sm = format_session_memory(self._cur_session_id)
                if sm:
                    parts.append(sm)
            except Exception as e:
                print(f"[Prompt] Session memory error: {e}")

        # 3-6. Retrieved memory blocks — dedup across blocks + token budget
        #      (same fact used to be injected 2-3x via different blocks)
        MEM_BUDGET_TOKENS = 1500
        mem_seen: set = set()
        mem_used = 0

        def _mem_block(header: str, lines: list, seen: set, budget: int) -> None:
            nonlocal mem_used
            kept = []
            for ln in lines:
                if not ln or not ln.strip():
                    continue
                sig = re.sub(r"^\s*\[[^\]]*\]\s*", "", ln)[:150]
                if sig in seen:
                    continue
                toks = max(1, len(ln) // 4)
                if mem_used + toks > budget:
                    break
                seen.add(sig)
                mem_used += toks
                kept.append(ln)
            if kept:
                parts.append("\n".join([header] + kept))

        _trivial = self._is_trivial_query(query)

        # 3. Retrieved memories (hybrid search across memory_items)
        try:
            if query and not _trivial:
                retrieved = hybrid_search(query, limit=5)
                lines = []
                for r in retrieved:
                    ctype = r.get("type", "memory")
                    content = r.get("content", "")
                    if content:
                        lines.append(f"  [{ctype}] {content[:300]}")
                _mem_block("[RETRIEVED MEMORIES — relevant past context]", lines, mem_seen, MEM_BUDGET_TOKENS)
        except Exception as e:
            print(f"[Prompt] Retrieval error: {e}")

        # 4. ChromaDB auto-retrieval (vector search across long-term facts + session summaries)
        try:
            if not hasattr(self, '_chroma') or not self._chroma:
                from memory.chroma_memory import ChromaMemory, CHROMA_VECTOR_SEARCH_ENABLED
                if CHROMA_VECTOR_SEARCH_ENABLED:
                    self._chroma = ChromaMemory()
            if self._chroma and query and not _trivial:
                chroma_results = self._chroma.search_memory(query, n_results=5)
                lines = []
                for r in chroma_results:
                    val = r.get("value", "")[:200]
                    cat = r.get("category", "")
                    key = r.get("key", "")
                    if val:
                        lines.append(f"  [{cat}/{key}] {val}")
                _mem_block("[LONG-TERM FACTS — ChromaDB vector search]", lines, mem_seen, MEM_BUDGET_TOKENS)
        except Exception as e:
            print(f"[Prompt] ChromaDB retrieval error: {e}")

        # 5. Core memory (ChromaDB facts — identity, preferences, projects, notes)
        try:
            from memory.memory_manager import format_core_memory
            core = format_core_memory()
            if core:
                parts.append(core)
        except Exception as e:
            print(f"[Prompt] Core memory error: {e}")

        # 6. Long-term memory items (from SQLite memory_items table)
        try:
            ltm = format_memory_items_from_sqlite()
            if ltm:
                _mem_block("[LONG-TERM MEMORY ITEMS]", ltm.split("\n"), mem_seen, MEM_BUDGET_TOKENS)
        except Exception as e:
            print(f"[Prompt] Long-term memory error: {e}")

        # 7. Timeline (relevant milestones — semantic search based on current context)
        try:
            if query:
                tl_events = search_timeline(query, limit=10)
                if tl_events:
                    tl_lines = ["[TIMELINE — relevant historical events]"]
                    for ev in tl_events:
                        ts = ev.get('timestamp', '')[:10]
                        desc = ev.get('description', '')
                        imp = ev.get('importance', 0.5)
                        marker = " ★" if imp >= 0.8 else ""
                        tl_lines.append(f"  {ts} — {desc}{marker}")
                    parts.append("\n".join(tl_lines))
            else:
                tl = format_timeline(limit=15)
                if tl:
                    parts.append(tl)
        except Exception as e:
            print(f"[Prompt] Timeline error: {e}")

        # 8. Active project context
        try:
            if self._cur_project_id and self._cur_project_id != "default_project":
                proj = get_project(self._cur_project_id)
                if proj:
                    proj_lines = [f"[ACTIVE PROJECT — {proj.get('name', 'Unknown')}]"]
                    from memory.conversation_db import search_memory_items
                    proj_mems = search_memory_items(proj.get('name', ''), limit=5)
                    for m in proj_mems:
                        content = (m.get('content') or '')[:200]
                        if content:
                            proj_lines.append(f"  {content}")
                    parts.append("\n".join(proj_lines))
        except Exception as e:
            print(f"[Prompt] Project context error: {e}")

        # 9. Tool memory (current session tool states)
        try:
            if self._cur_session_id:
                from memory.conversation_db import get_tool_memory
                known_tools = ["browser", "python", "vision", "git", "terminal", "file_controller", "code_helper"]
                ts_lines = []
                for tname in known_tools:
                    states = get_tool_memory(tname, self._cur_session_id)
                    if states:
                        ts_lines.append(f"  [{tname}]")
                        for k, v in states.items():
                            ts_lines.append(f"    {k}: {v[:100]}")
                if ts_lines:
                    ts_lines.insert(0, "[TOOL STATES — remembered tool context]")
                    parts.append("\n".join(ts_lines))
        except Exception as e:
            print(f"[Prompt] Tool state error: {e}")

        # 10. Knowledge graph (entities related to current query)
        try:
            if query and not _trivial:
                from memory.entity_store import get_entity_graph
                kg = get_entity_graph(query)
                if kg:
                    parts.append(kg)
        except Exception as e:
            print(f"[Prompt] Knowledge graph error: {e}")

        parts.append(time_ctx)
        parts.append(
            "[CONTINUOUS TOOL EXECUTION MODE]\n"
            "When doing multi-step tasks, keep calling tools "
            "in a continuous loop using tool results to decide the next step.\n"
            "RULE: Do NOT stop mid-task to ask the user 'what to do next'. "
            "Each tool result should tell you what to do next. "
            "Keep the loop going until the task is fully complete or user says stop."
        )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Speaking state & TTS
    # ------------------------------------------------------------------

    @staticmethod
    def _is_trivial_query(query: str) -> bool:
        """Retrieval gating (self-RAG style): skip memory/context retrieval
        for greetings and 1-2 word queries — saves tokens and avoids
        context distraction."""
        q = (query or "").strip().lower()
        if len(q) < 6:
            return True
        if re.fullmatch(r"(hi+|hey+|hello+|yo+|ok+|okay+|thanks|thank you|nice|great|good)\W*", q):
            return True
        return False

    def _tts_worker(self) -> None:
        self._tts_ready.wait(timeout=120)

        def _synth_loop():
            while True:
                text = self._tts_queue.get()
                if self._should_stop.is_set():
                    self._should_stop.clear()
                    self._tts_queue.task_done()
                    continue
                try:
                    streaming = bool(getattr(self._tts, "supports_streaming", False))
                    if streaming and text and self._tts:
                        for chunk in self._tts.synthesize_stream(text):
                            if self._should_stop.is_set():
                                break
                            self._audio_queue.put(chunk)
                    else:
                        audio = self._tts.synthesize(text) if text and self._tts else np.array([], dtype=np.float32)
                        if self._should_stop.is_set():
                            self._should_stop.clear()
                            self._tts_queue.task_done()
                            continue
                        self._audio_queue.put(audio)
                except Exception as e:
                    print(f"[TTS] synth error: {e}")
                self._audio_queue.put(None)

        threading.Thread(target=_synth_loop, daemon=True).start()

        while True:
            try:
                item = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._should_stop.is_set():
                self._should_stop.clear()
                continue
            if item is None:
                with self._speaking_lock:
                    self._speaking = False
                self._tts_queue.task_done()
                if self._tts_queue.empty() and not self.ui.muted:
                    self.ui.set_state("LISTENING")
                continue
            streaming = bool(getattr(self._tts, "supports_streaming", False))
            if streaming and isinstance(item, bytes):
                self._play_stream_chunk(item)
            else:
                self._play_blocking(item)

    def _play_stream_chunk(self, data: bytes) -> None:
        """Write one raw PCM chunk to the streaming output device."""
        if self._out_stream is None:
            if not self._open_out_stream():
                return
            with self._speaking_lock:
                self._speaking = True
            self.ui.set_state("SPEAKING")
        try:
            self._out_stream.write(np.frombuffer(data, dtype=np.int16))
        except Exception as e:
            print(f"[TTS] playback error: {e}")
            self._close_out_stream()

    def _open_out_stream(self) -> bool:
        try:
            self._out_stream = sd.OutputStream(
                samplerate=self._tts.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=2400,
            )
            self._out_stream.start()
            return True
        except Exception as e:
            print(f"[TTS] output stream error: {e}")
            self._out_stream = None
            return False

    def _close_out_stream(self) -> None:
        if self._out_stream is not None:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None

    def _play_blocking(self, audio: np.ndarray) -> None:
        """Legacy playback path (non-streaming engines: Edge, Kokoro, ...)."""
        if audio.size > 0:
            with self._speaking_lock:
                self._speaking = True
            self.ui.set_state("SPEAKING")
            sd.play(audio, self._tts.sample_rate)
            duration = len(audio) / self._tts.sample_rate
            time.sleep(duration)
        with self._speaking_lock:
            self._speaking = False

    def set_speaking(self, value: bool) -> None:
        with self._speaking_lock:
            self._speaking = value
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def speak(self, text: str) -> None:
        if not text:
            return
        # Gemini Live mode speaks natively through the persistent session
        _live = getattr(self, "_live_provider", None)
        if _live is not None and _live.session_active:
            _live.speak(text)
            return
        if not self._tts:
            return
        with self._speaking_lock:
            self._speaking = True
        self._tts_queue.put(text)

    def speak_error(self, tool_name: str, error) -> None:
        short = str(error)[:2000]
        print(f"ERR: {tool_name} — {short}")
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"{tool_name} encountered an error.")

    def _on_live_output(self, text: str) -> None:
        """Model output transcription from a voice-initiated Live turn."""
        try:
            self._extract_and_log(text)
        except Exception:
            pass
        try:
            self._deliver_response(text, source="live")
        except Exception:
            pass

    def _on_live_user(self, text: str) -> None:
        """User speech transcription from the Live session (logging only)."""
        self.ui.write_log(f"You: {text}")
        try:
            from memory.entity_store import link_fact
            link_fact("user_message", "input", text[:500])
        except Exception:
            pass

    def _on_live_speaking(self, value: bool) -> None:
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def stop_speaking(self) -> None:
        sd.stop()
        self._close_out_stream()
        self._should_stop.set()
        with self._speaking_lock:
            self._speaking = False
        # Gemini Live mode: drain the persistent session's audio queue too
        _live = getattr(self, "_live_provider", None)
        if _live is not None:
            try:
                _live.interrupt()
            except Exception:
                pass
        # Regular TTS engines: interrupt in-flight synthesis as well
        if (_live is None or not getattr(_live, "session_active", False)) and self._tts is not None:
            try:
                self._tts.interrupt()
            except Exception:
                pass
        while not self._tts_queue.empty():
            try:
                self._tts_queue.get_nowait()
                self._tts_queue.task_done()
            except queue.Empty:
                break
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        # Hashim Live TTS is a blocking engine — clear the stop flag so the
        # next response is not dropped (the interrupted turn already stopped)
        if self._config.get("tts_engine", "").lower() == "hashim_live":
            self._should_stop.clear()
        self.ui.set_state("LISTENING")

    def _signal_cancel(self) -> None:
        with self._cancel_lock:
            self._cancel_seq += 1
            self._cancel_event.set()

    def _is_cancelled(self, my_seq: int) -> bool:
        with self._cancel_lock:
            return self._cancel_seq != my_seq

    def _stop_and_listen(self) -> None:
        self.stop_speaking()
        if not self.ui.muted:
            self.ui.set_state("LISTENING")

    # ------------------------------------------------------------------
    # Live reconfigure (called when user clicks Apply in Configure panel)
    # ------------------------------------------------------------------

    def reconfigure(self, new_config: dict) -> None:
        threading.Thread(
            target=self._do_reconfigure, args=(new_config,), daemon=True
        ).start()

    def _do_reconfigure(self, new_config: dict) -> None:
        old_stt_engine = self._config.get("stt_engine", "whisper").lower()
        old_llm_model  = self._config.get("llm_model", "")
        new_stt_engine = new_config.get("stt_engine", "whisper").lower()
        self._config = new_config

        try:
            from core.installer import install_for_config
            install_for_config(new_config, log=self.ui.write_log)
        except Exception as e:
            self.ui.write_log(f"ERR: Dependency install — {e}")

        try:
            from core.tts import create_tts_player
            self._tts = create_tts_player(new_config)
            self._tts_ready.set()
            self.ui.write_log("SYS: TTS reconfigured.")
        except Exception as e:
            self.ui.write_log(f"ERR: TTS reconfigure — {e}")

        if old_stt_engine == new_stt_engine:
            try:
                stt_language = new_config.get("stt_language", "auto")
                if new_stt_engine == "vosk":
                    from core.stt import VoskSTT
                    self._stt = VoskSTT(new_config.get("vosk_model_path"), language=stt_language)
                elif new_stt_engine == "gemini":
                    from core.stt import GeminiSTT
                    gemini_key = new_config.get("gemini_voice_api_key", "").strip()
                    if not gemini_key:
                        raise RuntimeError(
                            "Gemini STT requires 'gemini_voice_api_key' — "
                            "set it in Configure → Text-to-Speech → Voice API Key."
                        )
                    self._stt = GeminiSTT(gemini_key, language=stt_language)
                elif new_stt_engine == "groq":
                    from core.stt import GroqSTT
                    groq_key = new_config.get("groq_api_key", "")
                    self._stt = GroqSTT(groq_key, language=stt_language)
                else:
                    from core.stt import WhisperSTT
                    self._stt = WhisperSTT(new_config.get("stt_model", "base"), language=stt_language)
                self.ui.write_log("SYS: STT reconfigured.")
            except Exception as e:
                self.ui.write_log(f"ERR: STT reconfigure — {e}")
        else:
            self.ui.write_log("SYS: STT engine changed — restart required.")

        # Save config to disk
        try:
            API_CONFIG_PATH.write_text(json.dumps(new_config, indent=4), encoding="utf-8")
        except Exception as e:
            self.ui.write_log(f"ERR: Config save — {e}")

        if new_config.get("llm_model", "") != old_llm_model:
            self.ui.write_log("SYS: Warming up new LLM model…")
            from core.llm_client import warmup_model
            warmup_model()
            self.ui.write_log("SYS: New LLM model ready.")

                # Re-wire Gemini Live callbacks if provider mode changed
        try:
            if new_config.get("llm_provider", "") == "gemini_live":
                from core.llm_provider import get_registry
                _live = get_registry().get_provider(
                    "gemini_live",
                    api_key=new_config.get("gemini_api_key", ""),
                    model=new_config.get("gemini_live_model", ""),
                )
                _live.register_tool_handler(self._execute_tool)
                _live.register_output_cb(self._on_live_output)
                _live.register_user_cb(self._on_live_user)
                _live.register_muted_getter(lambda: self.ui.muted)
                _live.register_speaking_cb(self._on_live_speaking)
                _live.set_system_prompt(_load_system_prompt())
                self._live_provider = _live
            else:
                if self._live_provider is not None:
                    try:
                        self._live_provider.stop()
                    except Exception:
                        pass
                self._live_provider = None
        except Exception as e:
            print(f"[Live] Callback re-wire failed: {e}")

        if old_stt_engine == new_stt_engine:
            self.speak("Configuration applied.")
        else:
            self.speak("LLM and TTS updated. Restart for speech engine change.")

    # ------------------------------------------------------------------
    # Text command (from UI input box)
    # ------------------------------------------------------------------

    def _on_text_command(self, text: str) -> None:
        self.stop_speaking()
        self._signal_cancel()
        self._text_queue.put(text)

    # ------------------------------------------------------------------
    # Inline chat actions (F7: Regenerate / Continue)
    # ------------------------------------------------------------------

    def _handle_chat_action(self, action: str) -> None:
        """Handle action:// chips from the chat log.

        Runs on the GUI thread (anchor click) — so it only *queues* the work;
        the text-pipeline worker performs all conversation mutation safely.
        """
        try:
            self._signal_cancel()
            self._text_queue.put(("action_msg", action))
        except Exception as e:
            print(f"[ChatAction] {action} queue error: {e}")

    def _run_chat_action(self, action: str) -> None:
        """Worker-side execution of a chat action (regenerate/continue)."""
        try:
            if action == "regenerate":
                # Drop the trailing assistant/tool turns and re-send the
                # newest user message for a fresh answer.
                while self._conversation and self._conversation[-1].get("role") in (
                        "assistant", "tool"):
                    self._conversation.pop()
                last_user = next(
                    (m["content"] for m in reversed(self._conversation)
                     if m.get("role") == "user"), None)
                if not last_user:
                    self.ui.write_log("SYS: Nothing to regenerate yet.")
                    return
                self.ui.write_log("SYS: ↻ Regenerating last answer…")
                self._process_message_inner(last_user, source="ui", extra_meta=None)
            elif action == "continue":
                last_ai = next(
                    (m["content"] for m in reversed(self._conversation)
                     if m.get("role") == "assistant"), None)
                if not last_ai:
                    self.ui.write_log("SYS: Nothing to continue yet.")
                    return
                self.ui.write_log("SYS: ▸ Continuing last answer…")
                self._process_message_inner(
                    "Continue your previous answer from where it stopped. "
                    "Do not repeat what you already wrote.",
                    source="ui", extra_meta=None)
            else:
                self.ui.write_log(f"SYS: Unknown chat action '{action}'.")
        except Exception as e:
            print(f"[ChatAction] {action} error: {e}")
            import traceback
            traceback.print_exc()

    def _on_text_command_with_files(self, text: str, extra: dict = None) -> None:
        """Handle text command with attached files (ChatGPT style).
        
        NOTE: stop_speaking() is NOT called here — this runs in a daemon thread
        and sounddevice access from non-main thread is unsafe.
        stop_speaking() is called later in _process_message_inner.
        """
        try:
            self._signal_cancel()
            self._text_queue.put(("file_msg", text, extra or {}))
        except Exception as e:
            print(f"[Orthos] ERROR in _on_text_command_with_files: {e}")
            import traceback
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Tool execution (routing unchanged from original)
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, args: dict) -> str:
        with Timer() as _timer:
            result = self._exec_tool_inner(name, args)
        elapsed = _timer.elapsed
        success = "failed" not in result.lower() and "error" not in result.lower()
        self._metrics.inc_tool_call(name, success=success)
        self._metrics.observe_request_duration("tool", name, elapsed)
        return result

    def _exec_tool_inner(self, name: str, args: dict) -> str:
        print(f"[Orthos] 🔧 {name}  {args}")
        self.ui.set_state("THINKING")

        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}},
                              session_id=self._cur_session_id,
                              project_id=self._cur_project_id)
                if hasattr(self, '_chroma') and self._chroma:
                    self._chroma.add_memory(category, key, value[:2000])
                print(f"[Memory] 💾 {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return "__SILENT__"

        if name == "forget_memory":
            from memory.memory_manager import forget
            result = forget(args.get("key", ""), args.get("category", "notes"))
            print(f"[Memory] 🗑️ {result}")
            return result

        if name == "search_memory":
            from memory.chroma_memory import search_all_memories
            query = args.get("query", "")
            limit_val = args.get("limit", 10)
            mode = args.get("mode", "summary")
            if isinstance(limit_val, str):
                try:
                    limit_val = int(limit_val)
                except ValueError:
                    limit_val = 10
            # Multi-signal fusion: BM25 + FTS5 + ChromaDB vector + entity graph
            chroma_result = search_all_memories(query, limit_val, mode=mode,
                                                project_id=self._cur_project_id or "")
            return chroma_result

        if name == "list_memories":
            from memory.memory_manager import format_memory_summary, clear_expired_memories, enforce_memory_cap
            _ = clear_expired_memories()
            _ = enforce_memory_cap()
            result = format_memory_summary()
            return result

        if name == "save_procedure":
            from memory.memory_manager import remember
            key = args.get("key", "")
            value = args.get("value", "")
            if key and value:
                result = remember(key, value, category="procedures",
                                  session_id=self._cur_session_id,
                                  project_id=self._cur_project_id)
                print(f"[Memory] 📋 Procedure: {key} = {value[:80]}")
            else:
                result = "Error: both key and value required"
            return result

        if name == "list_procedures":
            from memory.memory_manager import load_memory
            memory = load_memory()
            procs = memory.get("procedures", {})
            if not procs:
                return "No procedures stored yet."
            lines = ["[PROCEDURES — learned workflows and rules]"]
            for key, entry in procs.items():
                val = entry.get("value") if isinstance(entry, dict) else entry
                if val:
                    lines.append(f"  {key}: {val}")
            return "\n".join(lines)

        if name == "core_memory_append":
            from memory.memory_manager import core_memory_append as _cma
            return _cma(args.get("key", ""), args.get("value", ""),
                        session_id=self._cur_session_id,
                        project_id=self._cur_project_id)

        if name == "core_memory_replace":
            from memory.memory_manager import core_memory_replace as _cmr
            return _cmr(args.get("key", ""), args.get("value", ""),
                        session_id=self._cur_session_id,
                        project_id=self._cur_project_id)

        if name == "archival_memory_search":
            from memory.memory_manager import archival_memory_search as _ams
            query = args.get("query", "")
            limit_val = args.get("limit", 10)
            if isinstance(limit_val, str):
                try:
                    limit_val = int(limit_val)
                except ValueError:
                    limit_val = 10
            return _ams(query, limit_val)

        if name == "context_status":
            from memory.memory_manager import context_status as _cs
            return _cs()

        # ── Timeline tools ──
        if name == "search_timeline":
            query = args.get("query", "")
            limit_val = args.get("limit", 10)
            event_type = args.get("event_type", "")
            days_back = args.get("days_back", 0)
            if isinstance(limit_val, str):
                try: limit_val = int(limit_val)
                except ValueError: limit_val = 10
            if isinstance(days_back, str):
                try: days_back = int(days_back)
                except ValueError: days_back = 0
            results = search_timeline(query, limit=limit_val, event_type=event_type,
                                      project_id=self._cur_project_id or "", days_back=days_back)
            if not results:
                return f"No timeline events found for: {query}"
            lines = ["[TIMELINE SEARCH RESULTS]"]
            for ev in results:
                ts = ev.get('timestamp', '')[:10]
                desc = ev.get('description', '')
                etype = ev.get('event_type', 'note')
                imp = ev.get('importance', 0.5)
                marker = " ★" if imp >= 0.8 else ""
                lines.append(f"  [{ts}] ({etype}){marker} {desc}")
            return "\n".join(lines)

        if name == "save_timeline_event":
            event_type = args.get("event_type", "note")
            description = args.get("description", "")
            importance = args.get("importance", 0.5)
            if isinstance(importance, str):
                try: importance = float(importance)
                except ValueError: importance = 0.5
            if not description:
                return "Error: description is required"
            log_timeline_event(
                self._cur_session_id or "", event_type, description,
                auto=False, importance=importance,
                project_id=self._cur_project_id or "",
            )
            print(f"[Timeline] Event logged: {event_type} — {description[:80]}")
            return f"Timeline event saved: {event_type} — {description[:100]}"

        # ── Project tools ──
        if name == "list_projects":
            projects = list_projects()
            if not projects:
                return "No projects found. Use create_project to make one."
            lines = ["[PROJECTS]"]
            for p in projects:
                active_marker = " ← active" if p.get("id") == self._cur_project_id else ""
                lines.append(f"  {p['id']}: {p['name']}{active_marker}")
            return "\n".join(lines)

        if name == "create_project":
            pname = args.get("name", "")
            if not pname:
                return "Error: name is required"
            pid = create_project(pname)
            print(f"[Project] Created: {pid} — {pname}")
            return f"Project created: {pname} (id: {pid})"

        if name == "set_active_project":
            pid_or_name = args.get("project_id", "")
            if not pid_or_name:
                return "Error: project_id is required"
            # Resolve name → id if needed
            projects = list_projects()
            target_id = None
            for p in projects:
                if p["id"] == pid_or_name or p["name"].lower() == pid_or_name.lower():
                    target_id = p["id"]
                    break
            if not target_id:
                return f"Project not found: {pid_or_name}. Use list_projects to see available projects."
            self._cur_project_id = target_id
            if self._cur_session_id:
                set_session_project(self._cur_session_id, target_id)
            print(f"[Project] Switched to: {target_id}")
            return f"Switched to project: {target_id}"

        if name == "get_project_memories":
            pid_or_name = args.get("project_id", "")
            if not pid_or_name:
                return "Error: project_id is required"
            projects = list_projects()
            target_id = None
            target_name = ""
            for p in projects:
                if p["id"] == pid_or_name or p["name"].lower() == pid_or_name.lower():
                    target_id = p["id"]
                    target_name = p["name"]
                    break
            if not target_id:
                return f"Project not found: {pid_or_name}"
            # Get project-scoped memory items
            from memory.conversation_db import search_memory_items
            mems = search_memory_items(target_name, limit=10)
            lines = [f"[PROJECT MEMORIES — {target_name}]"]
            for m in mems:
                content = (m.get("content") or "")[:300]
                lines.append(f"  • {content}")
            # Get project timeline
            tl = format_timeline(limit=10, project_id=target_id)
            if tl:
                lines.append(tl)
            return "\n".join(lines)

        # ── Tool memory ──
        if name == "save_tool_state":
            tool_name = args.get("tool_name", "")
            key = args.get("key", "")
            value = args.get("value", "")
            if not all([tool_name, key, value]):
                return "Error: tool_name, key, and value are all required"
            set_tool_memory(tool_name, self._cur_session_id or "", key, value)
            print(f"[ToolMemory] {tool_name}/{key} = {value[:80]}")
            return f"Tool state saved: {tool_name}/{key}"

        if name == "get_tool_state":
            tool_name = args.get("tool_name", "")
            if not tool_name:
                return "Error: tool_name is required"
            from memory.conversation_db import get_tool_memory
            states = get_tool_memory(tool_name, self._cur_session_id or "")
            if not states:
                return f"No saved state for tool: {tool_name}"
            lines = [f"[TOOL STATE — {tool_name}]"]
            for k, v in states.items():
                lines.append(f"  {k}: {v}")
            return "\n".join(lines)

        # ── GraphRAG ──
        if name == "search_knowledge_graph":
            query = args.get("query", "")
            max_hops = args.get("max_hops", 2)
            if isinstance(max_hops, str):
                try: max_hops = int(max_hops)
                except ValueError: max_hops = 2
            if not query:
                return "Error: query is required"
            from memory.entity_store import get_entity_graph, graph_retrieve
            # First try graph_retrieve (multi-hop with full context)
            graph_results = graph_retrieve(query, max_hops=max_hops)
            if graph_results:
                lines = ["[KNOWLEDGE GRAPH — entity relationships]"]
                seen = set()
                for r in graph_results:
                    sig = (r.get("entity", ""), r.get("relation", ""), r.get("connected_entity", ""))
                    if sig not in seen:
                        seen.add(sig)
                        lines.append(f"  [{r.get('entity_type', '?')}] {r.get('entity', '')} "
                                     f"--[{r.get('relation', '?')}]--> "
                                     f"[{r.get('connected_type', '?')}] {r.get('connected_entity', '')}")
                        obs = r.get("observations", "")
                        if obs:
                            lines.append(f"    {obs[:200]}")
                return "\n".join(lines)
            # Fallback: format entity graph
            kg = get_entity_graph(query)
            return kg or f"No entities found for: {query}"

        result = "Done."
        try:
            if name == "open_app":
                r = open_app(parameters=args, response=None, player=self.ui)
                result = r or f"Opened {args.get('app_name')}."

            elif name == "weather_report":
                r = weather_action(parameters=args, player=self.ui)
                result = r or "Weather delivered."

            elif name == "file_controller":
                r = file_controller(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "send_message":
                r = send_message(parameters=args, response=None, player=self.ui, session_memory=None)
                result = r or f"Message sent to {args.get('receiver')}."

            elif name == "reminder":
                r = reminder(parameters=args, response=None, player=self.ui)
                result = r or "Reminder set."

            elif name == "youtube_video":
                r = youtube_video(parameters=args, response=None, player=self.ui)
                result = r or "Done."

            elif name == "screen_process":
                r = screen_process(parameters=args, response=None, player=self.ui, session_memory=self._conversation[-10:])
                result = r if isinstance(r, str) and r else "Screen analyzed."

            elif name == "screen_locate":
                r = screen_locate(parameters=args, player=self.ui, session_memory=self._conversation[-10:])
                result = r if isinstance(r, str) and r else "Element not found."

            elif name == "computer_settings":
                r = computer_settings(parameters=args, response=None, player=self.ui)
                result = r or "Done."

            elif name == "desktop_control":
                r = desktop_control(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "code_helper":
                r = code_helper(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."

            elif name == "dev_agent":
                r = dev_agent(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."

            elif name == "agent_task":
                from agent.task_queue import get_queue, TaskPriority
                priority_map = {
                    "low": TaskPriority.LOW,
                    "normal": TaskPriority.NORMAL,
                    "high": TaskPriority.HIGH,
                }
                priority = priority_map.get(
                    args.get("priority", "normal").lower(), TaskPriority.NORMAL
                )
                task_id = get_queue().submit(
                    goal=args.get("goal", ""), priority=priority, speak=self.speak
                )
                result = f"Task started (ID: {task_id})."

            elif name == "web_search":
                self.ui.set_state("PROCESSING")
                r = web_search_action(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "webfetch":
                self.ui.set_state("PROCESSING")
                r = webfetch_action(parameters=args, player=self.ui)
                result = r or "Could not fetch URL."

            elif name == "file_processor":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                r = file_processor(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."

            elif name == "computer_control":
                r = computer_control(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "run_terminal":
                r = run_terminal(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "game_updater":
                r = game_updater(parameters=args, player=self.ui, speak=self.speak)
                result = r or "Done."

            elif name == "flight_finder":
                r = flight_finder(parameters=args, player=self.ui)
                result = r or "Done."

            elif name == "list_mcp_servers":
                if not self._mcp_manager:
                    return "MCP manager not initialized."
                config = self._mcp_manager.get_all_servers_config()
                lines = ["## MCP Servers\n"]
                for srv_name, cfg in config.items():
                    en = cfg.get("enabled", False)
                    status = self._mcp_manager.get_server_status(srv_name)
                    cat = cfg.get("category", "Uncategorized")
                    desc = cfg.get("description", "")
                    tools = [t for t in _tool_registry.list_tools()
                             if t["name"].startswith(f"mcp_{srv_name}_")]
                    icon = "🟢" if en and status == "running" else "🔴" if en else "⚪"
                    lines.append(
                        f"{icon} **{srv_name}** ({cat}) — {status} | {len(tools)} tools\n"
                        f"   {desc}\n"
                        f"   Load all tools: `get_tool_definition(tool_name=\"{srv_name}\")`\n"
                    )
                lines.append(
                    "\nTo use a tool: call `search_tools(query)` to find it, "
                    "then `get_tool_definition(tool_name='TOOL_NAME')` to load its full schema.\n"
                    "To load ALL tools from a server at once: "
                    "`get_tool_definition(tool_name='SERVER_NAME')`."
                )
                return "\n".join(lines)

            elif name == "manage_mcp_server":
                if not self._mcp_manager:
                    return "MCP manager not initialized."
                action = str(args.get("action", "")).strip().lower()
                server = str(args.get("server", "")).strip()
                config = self._mcp_manager.get_all_servers_config()
                if action == "install":
                    package = str(args.get("package", "")).strip()
                    if not package:
                        return "Tell me the MCP package to install, for example @modelcontextprotocol/server-github."
                    generated_name = server or package.split("/")[-1].replace("server-", "").replace("-mcp", "")
                    command = str(args.get("command") or "npx").strip().lower()
                    if command not in ("npx", "uvx"):
                        return "For safety, MCP installation supports npx or uvx only."
                    install_config = {
                        "enabled": True,
                        "category": str(args.get("category") or "Development"),
                        "command": command,
                        "args": (["-y", package] if command == "npx" else [package]),
                        "description": f"Installed by Orthos: {package}",
                    }
                    return self._mcp_manager.install_server(generated_name, install_config)
                if not server or server not in config:
                    available = ", ".join(sorted(config)) or "none"
                    return f"I couldn't find '{server}'. Available MCP servers: {available}."
                if action == "start":
                    return self._mcp_manager.start_server(server)
                if action == "stop":
                    return self._mcp_manager.stop_server(server)
                if action == "restart":
                    return self._mcp_manager.restart_server(server)
                if action == "enable":
                    if config[server].get("enabled", False):
                        return self._mcp_manager.start_server(server)
                    return self._mcp_manager.toggle_server(server)
                if action == "disable":
                    if not config[server].get("enabled", False):
                        return self._mcp_manager.stop_server(server)
                    return self._mcp_manager.toggle_server(server)
                return "Supported MCP actions are: start, stop, restart, enable, disable, and install."

            elif name == "shutdown_orthos":
                self.ui.write_log("SYS: Shutdown requested.")

                def _shutdown():
                    self.speak("Goodbye.")
                    time.sleep(2.5)
                    _os._exit(0)

                threading.Thread(target=_shutdown, daemon=True).start()
                return "Shutting down."

            elif name.startswith("mcp_"):
                if self._mcp_manager is not None:
                    try:
                        if self._mcp_manager.is_write_tool(name) and self._mcp_manager.confirm_write():
                            self.ui.write_log(f"SYS: ⚠ MCP write action: {name}")
                    except Exception:
                        pass
                r = _tool_registry.execute(name, args)
                result = r or "Done."

            elif name in ("search_tools", "get_tool_definition"):
                fn = _tool_registry.get(name)
                if fn:
                    r = fn(parameters=args, response=None, player=self.ui)
                    result = r or "Done."
                else:
                    result = f"Meta-tool '{name}' not found in registry"

            else:
                # Try registry as fallback (handles dynamically registered tools)
                fn = _tool_registry.get(name)
                if fn:
                    r = fn(parameters=args, response=None, player=self.ui)
                    result = r or "Done."
                else:
                    result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        print(f"[Orthos] 📤 {name} → {str(result)[:80]}")
        return result

    # ------------------------------------------------------------------
    # LLM processing loop
    # ------------------------------------------------------------------

    def _process_message(self, user_text: str, source: str = "ui", extra_meta: dict | None = None) -> None:
        try:
            self._process_message_inner(user_text, source, extra_meta)
        except Exception as e:
            print(f"[Orthos] FATAL ERROR in _process_message: {e}")
            import traceback
            traceback.print_exc()

    def _process_message_inner(self, user_text: str, source: str, extra_meta: dict | None) -> None:
        self.stop_speaking()
        self._should_stop.clear()
        with self._cancel_lock:
            self._cancel_seq += 1
            _my_seq = self._cancel_seq
            self._cancel_event.clear()
        self.ui.set_state("THINKING")
        self.ui.write_log(f"You: {user_text}")

        self._conversation.append({"role": "user", "content": user_text})

        # Extract entities from user message for knowledge graph
        try:
            from memory.entity_store import link_fact
            link_fact("user_message", "input", user_text[:500])
        except Exception:
            pass

        for msg in self._conversation:
            msg.pop("images", None)

        # ── Budget: 75% input, 25% output. 2-way split (backup-compatible) ──
        self._conv_budget = 5000
        self._summary_rolling_budget = 15000
        self._summary_past_budget = 15000
        _tool_cost_est = 2500
        input_budget = 8192  # default (overridden inside try block below)
        try:
            cfg = _get_provider_config()
            from memory.conversation_db import get_model_context_window, _estimate_tokens
            ctx_window = get_model_context_window(cfg["model"], cfg["provider"])
            # All providers now standardized to 128K context window
            input_budget = int(ctx_window * 0.50)
            mcp_reserve = int(ctx_window * 0.25)
            _current_tools = _build_dynamic_tools()
            if _current_tools:
                _internal_tools = [t for t in _current_tools
                                   if not (t.get("function", {}).get("name", "")).startswith("mcp_")]
                _mcp_tools = [t for t in _current_tools
                              if (t.get("function", {}).get("name", "")).startswith("mcp_")]
                _tool_cost_est = _estimate_tokens(json.dumps(_internal_tools, ensure_ascii=False))
                _mcp_cost_est = _estimate_tokens(json.dumps(_mcp_tools, ensure_ascii=False))
            else:
                _tool_cost_est = 2500
                _mcp_cost_est = 0
            _base_core = _estimate_tokens(_load_system_prompt())
            _now = datetime.now()
            _base_core += _estimate_tokens(
                f"[CURRENT DATE & TIME]\nRight now it is: {_now.strftime('%A, %B %d, %Y — %I:%M %p')}\n"
                f"Use this to calculate exact times for reminders."
                f"\n\n[CONTINUOUS TOOL EXECUTION MODE]\nWhen doing multi-step tasks, keep calling tools "
                f"in a continuous loop using tool results to decide the next step.\n"
                f"RULE: Do NOT stop mid-task to ask the user 'what to do next'. "
                f"Each tool result should tell you what to do next. "
                f"Keep the loop going until the task is fully complete or user says stop."
            ) + 5000
            _remaining = input_budget - _tool_cost_est - _base_core
            if _remaining > 6000:
                self._summary_rolling_budget = int(_remaining * 0.20)
                self._summary_past_budget = int(_remaining * 0.20)
                self._conv_budget = int(_remaining * 0.55)
            if _mcp_cost_est > mcp_reserve:
                print(f"[Budget] WARN: MCP tools ({_mcp_cost_est:,} tok) exceed reserve ({mcp_reserve:,} tok)")
        except Exception:
            pass

        messages = [
            {"role": "system", "content": self._build_system_prompt()}
        ] + list(self._conversation)

        # ── Inbound media (e.g. Telegram images) → attach to latest user turn ──
        _attachment_content_ready = False
        if extra_meta:
            try:
                _img_files = extra_meta.get("image_files") or []
                if _img_files:
                    _inb64s = []
                    import PIL.Image as _PILImg
                    for _p in _img_files:
                        try:
                            if Path(_p).exists():
                                # Compress: resize to 720px max + JPEG 60% quality
                                _img = _PILImg.open(Path(_p))
                                if _img.mode in ("RGBA", "P"):
                                    _img = _img.convert("RGB")
                                _MAX_DIM = 720
                                _img.thumbnail((_MAX_DIM, _MAX_DIM), _PILImg.LANCZOS)
                                _buf = io.BytesIO()
                                _img.save(_buf, format="JPEG", quality=60, optimize=True)
                                _raw_bytes = _buf.getvalue()
                                _b64 = base64.b64encode(_raw_bytes).decode("ascii")
                                _inb64s.append(_b64)
                                _orig_kb = Path(_p).stat().st_size // 1024
                                _comp_kb = len(_raw_bytes) // 1024
                                print(f"[Orthos] Image compressed: {_p} ({_orig_kb}KB -> {_comp_kb}KB, {len(_b64)} chars)")
                            else:
                                print(f"[Orthos] WARNING: Image file not found: {_p}")
                        except Exception as e:
                            print(f"[Orthos] ERROR encoding image {_p}: {e}")
                    if _inb64s:
                        for _i in range(len(messages) - 1, -1, -1):
                            if messages[_i].get("role") == "user":
                                messages[_i] = {**messages[_i], "images": _inb64s, "image_files": list(_img_files)}
                                print(f"[Orthos] Images attached to user message")
                                break
                    else:
                        print(f"[Orthos] WARNING: No images could be encoded from {_img_files}")
                # Scanned PDFs have no text layer. The UI renders their pages and
                # supplies them to the vision model directly - never via a viewer or
                # repeated screenshots.
                _document_images = extra_meta.get("document_images") or []
                if _document_images:
                    for _i in range(len(messages) - 1, -1, -1):
                        if messages[_i].get("role") == "user":
                            _existing_images = list(messages[_i].get("images") or [])
                            messages[_i] = {
                                **messages[_i],
                                "images": _existing_images + list(_document_images),
                                "image_files": list(messages[_i].get("image_files") or ["rendered-pdf-pages"]),
                            }
                            _attachment_content_ready = True
                            print(f"[Orthos] Attached {len(_document_images)} rendered PDF page(s) to vision context")
                            break

                _attached = extra_meta.get("files") or []
                _non_img = [p for p in _attached if p not in _img_files]
                if _non_img:
                    _note = (chr(10) + chr(10) + "[User attached file(s): " + "; ".join(_non_img)
                             + " — use the file_processor or file_controller tools "
                               "to read/analyze them if needed.]")
                    for _i in range(len(messages) - 1, -1, -1):
                        if messages[_i].get("role") == "user":
                            messages[_i]["content"] = messages[_i].get("content", "") + _note
                            print(f"[Orthos] Non-image files attached to user message")
                            break

                # The UI pre-reads a bounded local preview for documents/data files.
                # Put that context on the user turn so analysis does not depend solely
                # on the model deciding to call a file tool first.
                _contexts = extra_meta.get("attachment_context") or []
                _context_parts = []
                _context_budget = 24000
                for _ctx in _contexts:
                    if not isinstance(_ctx, dict):
                        continue
                    _name = str(_ctx.get("name") or "attachment")
                    _kind = str(_ctx.get("type") or "file")
                    _detail = str(_ctx.get("detail") or "Ready to analyze")
                    _preview = str(_ctx.get("preview") or "").strip()
                    if _preview and _context_budget > 0:
                        _preview = _preview[:_context_budget]
                        _context_budget -= len(_preview)
                        _context_parts.append(
                            f"[Pre-read {_kind} attachment: {_name} — {_detail}]\n{_preview}"
                        )
                    else:
                        _context_parts.append(f"[Attachment metadata: {_kind} {_name} — {_detail}]")
                if _context_parts:
                    _attachment_context_note = "\n\n" + "\n\n".join(_context_parts)
                    for _i in range(len(messages) - 1, -1, -1):
                        if messages[_i].get("role") == "user":
                            messages[_i]["content"] = messages[_i].get("content", "") + _attachment_context_note
                            print(f"[Orthos] Added pre-read context for {len(_context_parts)} attachment(s)")
                            _attachment_content_ready = _attachment_content_ready or any(
                                bool(str(_ctx.get("preview") or "").strip())
                                for _ctx in _contexts if isinstance(_ctx, dict)
                            )
                            break
                if _attachment_content_ready:
                    messages[0]["content"] += (
                        "\n\n[ATTACHMENT HANDLING]\n"
                        "The attached document has already been extracted or rendered into this conversation. "
                        "Answer from that context. Do not open the file, capture the screen, or repeatedly invoke "
                        "file tools for this attachment."
                    )
            except Exception as e:
                print(f"[Orthos] ERROR processing extra_meta: {e}")
                import traceback
                traceback.print_exc()

        if self._cur_session_id:
            try:
                from memory.conversation_db import estimate_messages_tokens, summarize_turns, _estimate_tokens
                non_system = [m for m in messages if m.get("role") != "system"]
                raw_tokens = estimate_messages_tokens(non_system)
                # ── Partial trim (2026-09-06 policy) ────────────────────────
                # When the raw conversation crosses the 50k trigger, ONLY the
                # oldest ~15k are summarized into the rolling summary; the
                # newest turns stay RAW, so recent context keeps full fidelity.
                # (Old behaviour summarized the ENTIRE conversation, which
                # caused the model to "forget" the turn it was answering.)
                if raw_tokens > _RAW_CONV_TRIGGER_TOKENS and len(non_system) > 2:
                    current_msg = non_system.pop()          # latest message stays raw
                    keep_msg  = non_system.pop()            # and the previous one
                    if not non_system:                      # tiny chat: nothing to trim
                        non_system = [keep_msg, current_msg]
                    else:
                        # Save ALL unsaved turns first — the DB is the
                        # authoritative full history; nothing gets dropped.
                        unsaved = self._conversation[self._conv_saved_idx:]
                        if unsaved:
                            save_conversation(unsaved, self._cur_session_id)
                            # Keep sidebar info (turns/tokens) in sync after the save
                            try:
                                self.ui._win._session_refresh_sig.emit()
                            except Exception:
                                pass
                        trim_idx = _oldest_trim_split(non_system, _TRIM_CHUNK_TOKENS)
                        oldest = [non_system[i] for i in trim_idx]
                        rest   = [m for i, m in enumerate(non_system) if i not in set(trim_idx)]
                        summary = summarize_turns(oldest) if oldest else None
                        if summary:
                            previous = self._rolling_summary
                            if previous:
                                self._rolling_summary = f"[Earlier context]\n{previous}\n\n[Recent summarized context]\n{summary}"
                            else:
                                self._rolling_summary = summary
                            # Dynamic cap: rolling summary must not itself blow the budget
                            if _estimate_tokens(self._rolling_summary) > self._summary_rolling_budget:
                                condensed = summarize_turns([
                                    {"role": "user", "content":
                                     f"Condense this accumulated conversation summary into key points. "
                                     f"Keep it under {self._summary_rolling_budget} tokens:\n{self._rolling_summary}"},
                                ], target_tokens=self._summary_rolling_budget)
                                if condensed and _estimate_tokens(condensed) < self._summary_rolling_budget:
                                    self._rolling_summary = condensed
                                else:
                                    words = self._rolling_summary.split()
                                    self._rolling_summary = " ".join(words[-int(self._summary_rolling_budget / 1.3):])
                            set_session_memory(self._cur_session_id, "_rolling_summary", self._rolling_summary)
                            # NOTE: no local 'import time' here — it would make
                            # 'time' function-local and crash the tool-timing
                            # path below with UnboundLocalError. Module-level
                            # import (main.py top) is authoritative.
                            set_session_memory(self._cur_session_id, f"_summary_{int(time.time())}", summary)
                        # Rebuild: trimmed oldest removed, recent turns + current stay raw.
                        self._conversation = rest + [keep_msg, current_msg]
                        self._conv_saved_idx = len(self._conversation)
                        messages = [messages[0]] + list(self._conversation)
                        print(f"[Trim] partial: raw={raw_tokens}, summarized oldest {len(trim_idx)} msgs "
                              f"(~{_TRIM_CHUNK_TOKENS} tok), kept {len(rest)} recent msgs raw + current")
            except Exception as _e:
                print(f"[Trim] Conversation trim failed: {_e}")

        # ── Post-build assertion: total input must not exceed 75% budget ──
        try:
            _total_input = _estimate_tokens(json.dumps(messages, ensure_ascii=False))
            _total_input += _tool_cost_est
            if _total_input > input_budget:
                _excess = _total_input - input_budget
                print(f"[Budget] WARN: total_input ({_total_input:,}) > input_budget ({input_budget:,}) by {_excess:,} tok")
                # Force aggressive trim: shrink conv_budget by excess
                self._conv_budget = max(2048, self._conv_budget - _excess)
                # Re-trigger trim if conv_budget changed significantly
                if _excess > 2000 and self._cur_session_id:
                    non_system = [m for m in messages if m.get("role") != "system"]
                    if non_system and len(non_system) > 2:
                        current_msg = non_system.pop()
                        summary = summarize_turns(non_system)
                        if summary:
                            previous = self._rolling_summary
                            if previous:
                                self._rolling_summary = f"[Earlier context]\n{previous}\n\n[Recent conversation]\n{summary}"
                            else:
                                self._rolling_summary = summary
                        # A4 fix: save unsaved turns BEFORE dropping — otherwise
                        # a force-trim silently discards history from the DB.
                        unsaved = self._conversation[self._conv_saved_idx:]
                        if unsaved:
                            try:
                                save_conversation(unsaved, self._cur_session_id)
                                self.ui._win._session_refresh_sig.emit()
                            except Exception as _sve:
                                print(f"[Budget] Save-before-trim failed: {_sve}")
                        self._conversation = [current_msg]
                        self._conv_saved_idx = 1
                        messages = [messages[0]] + [current_msg]
                        print(f"[Budget] Force-trimmed to conv_budget={self._conv_budget}")
        except Exception:
            pass

        _NEEDS_LLM_ROUND = {"web_search", "webfetch", "screen_process", "screen_locate", "agent_task"}

        # Gemini Live mode speaks natively — never echo text through TTS
        _current_live_native = bool(getattr(self, "_live_provider", None))

        _round = 0
        _consecutive_no_tool = 0
        while True:
            _round += 1
            if self._is_cancelled(_my_seq):
                self._stop_and_listen()
                return

            # Skip auto screen capture if the user attached an image (e.g. via Telegram)
            _user_has_image = False
            for _m in reversed(messages):
                if _m.get("role") == "user":
                    _user_has_image = bool(_m.get("image_files"))
                    break

            if not _user_has_image and getattr(self.ui, "screen_vision", True):
                try:
                    _fresh, _ = _capture_screen_annotated(0, 0)
                    import PIL.Image as _PIL_Image
                    _img = _PIL_Image.open(io.BytesIO(_fresh))
                    # Resize to 720p (max width 1280) to reduce payload size
                    _MAX_W = 1280
                    if _img.width > _MAX_W:
                        _ratio = _MAX_W / _img.width
                        _new_h = int(_img.height * _ratio)
                        _img = _img.resize((_MAX_W, _new_h), _PIL_Image.LANCZOS)
                    _buf = io.BytesIO()
                    _img.save(_buf, format="JPEG", quality=_get_screenshot_quality("llm"))
                    _b64 = base64.b64encode(_buf.getvalue()).decode("ascii")
                    _cap_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    _cap_path = str(_CAPTURES_DIR / f"{_cap_ts}.jpg")
                    _img.save(_cap_path, format="JPEG", quality=_get_screenshot_quality("llm"))
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            messages[i] = {**messages[i], "images": [_b64], "image_files": [_cap_path]}
                            break
                except Exception as _e:
                    print(f"[Vision] Auto-fresh capture skipped: {_e}")

            final_content    = ""
            final_tool_calls: list = []
            _streamed: list[str] = []
            _spoke_sentences = False
            # Hashim Live TTS: buffer all sentences and speak the FULL output
            # at once for a more natural voice. Other TTS engines keep the
            # old sentence-by-sentence streaming flow.
            _is_hashim_tts = self._config.get("tts_engine", "").lower() == "hashim_live"
            _hashim_buffer: list[str] = []

            try:
                _current_tools = _build_dynamic_tools()
                if _attachment_content_ready:
                    _blocked_attachment_tools = {
                        "file_processor", "file_controller", "open_app",
                        "screen_process", "screen_locate",
                    }
                    _current_tools = [tool for tool in _current_tools if tool.get("function", {}).get("name") not in _blocked_attachment_tools]
                # ── Live streaming render (F1): tokens appear in the chat as
                # they arrive instead of one final card after the full reply.
                _stream_ui = getattr(self.ui, "_win", None)
                _stream_log = getattr(_stream_ui, "_log", None) if _stream_ui else None
                _stream_active = False
                # Stays True after streaming_end: the final card is already
                # in the log, so the deliver path must not log it again.
                _stream_rendered = False
                for event in call_llm_stream(messages, _current_tools, cancel_event=self._cancel_event):
                    if event["type"] == "sentence":
                        _streamed.append(event["text"])
                        # Live card update (skip for Gemini Live: its sentences
                        # are already written below and the card would dupe).
                        # Threadsafe: the LLM loop runs off the GUI thread.
                        if _stream_log is not None and not event.get("native_audio"):
                            if not _stream_active:
                                _stream_log.streaming_start_threadsafe()
                                _stream_active = True
                            _stream_log.streaming_append_threadsafe(event["text"])
                        # Gemini Live plays native audio — never re-speak via TTS/session
                        if not event.get("native_audio"):
                            if _is_hashim_tts:
                                # Buffer for full-output TTS
                                _hashim_buffer.append(event["text"])
                            else:
                                self.speak(event["text"])
                        else:
                            # Show streaming text for Gemini Live even though audio plays natively
                            self.ui.write_log(f"Orthos: {event['text']}")
                        _spoke_sentences = True
                    elif event["type"] == "done":
                        final_content    = event["content"]
                        final_tool_calls = event["tool_calls"]
                        # Finalize the live card with the authoritative text;
                        # the assembled card replaces the streaming draft.
                        if _stream_log is not None and _stream_active:
                            _stream_log.streaming_end_threadsafe(
                                final_content or "".join(_streamed))
                            _stream_active = False
                            _stream_rendered = True
                        # Hashim Live TTS: send the FULL response at once
                        if _is_hashim_tts and _hashim_buffer:
                            full_text = " ".join(_hashim_buffer).strip()
                            if full_text:
                                self.speak(full_text)
                            _hashim_buffer = []
                    elif event["type"] == "cancelled":
                        if _stream_log is not None and _stream_active:
                            _stream_log.streaming_abort_threadsafe()
                        self._stop_and_listen()
                        return
            except RuntimeError as e:
                if _stream_log is not None and _stream_active:
                    _stream_log.streaming_abort_threadsafe()
                    _stream_active = False
                if _QuotaExceededError is not None and isinstance(e, _QuotaExceededError):
                    print(f"ERR: {e}")
                    self.ui.write_log(f"ERR: {e}")
                    self.speak(str(e))
                elif _current_live_native:
                    # Gemini Live (Mark-L style): session errors are
                    # console-only — the provider reconnects on its own and
                    # the next turn works normally.
                    print(f"[GeminiLive] LLM round aborted: {e}")
                else:
                    self.speak_error("LLM", e)
                return

            if not final_tool_calls:
                if final_content:
                    assistant_msg = {"role": "assistant", "content": final_content}
                    messages.append(assistant_msg)
                    self._conversation.append(assistant_msg)
                    # Already rendered live (F1) or Gemini Live streamed it —
                    # logging again would duplicate the reply card.
                    _already_shown = _stream_rendered or (
                        _current_live_native and _spoke_sentences)
                    if not _already_shown:
                        self._deliver_response(final_content, source)
                    else:
                        self._deliver_response(final_content, source, quiet=True)
                    self._extract_and_log(final_content)
                    if not _spoke_sentences and not _current_live_native:
                        self.speak(final_content)
                break

            assistant_msg = {
                "role":       "assistant",
                "content":    final_content or "",
                "tool_calls": final_tool_calls,
            }
            messages.append(assistant_msg)
            self._conversation.append(assistant_msg)

            _only_memory = all(
                tc.get("function", {}).get("name") == "save_memory"
                for tc in final_tool_calls
            )
            if _only_memory and final_content:
                for tc in final_tool_calls:
                    fn    = tc.get("function", {})
                    targs = fn.get("arguments", {})
                    if isinstance(targs, str):
                        try:
                            targs = json.loads(targs)
                        except Exception:
                            targs = {}
                    self._execute_tool("save_memory", targs)
                assistant_msg2 = {"role": "assistant", "content": final_content}
                messages.append(assistant_msg2)
                self._conversation.append(assistant_msg2)
                if _stream_rendered or (_current_live_native and _spoke_sentences):
                    self._deliver_response(final_content, source, quiet=True)
                else:
                    self._deliver_response(final_content, source)
                self._extract_and_log(final_content)
                if not _spoke_sentences and not _current_live_native:
                    self.speak(final_content)
                break

            all_silent    = True
            _tool_results: list[tuple[str, str]] = []
            _has_image = False

            for tc in final_tool_calls:
                if self._is_cancelled(_my_seq):
                    self._stop_and_listen()
                    return

                fn    = tc.get("function", {})
                tname = fn.get("name", "")
                targs = fn.get("arguments", {})
                if isinstance(targs, str):
                    try:
                        targs = json.loads(targs)
                    except Exception:
                        targs = {}

                tc_id = tc.get("id", "")
                # Live tool progress: name + a compact one-line args preview
                # so the user sees WHAT runs, not just that something did.
                _args_preview = ""
                try:
                    _pv = ", ".join(f"{k}={str(v)[:40]!r}" for k, v in
                                    list(targs.items())[:2])
                    if _pv:
                        _args_preview = f"({_pv})"
                except Exception:
                    _args_preview = ""
                self.ui.write_log(f"SYS: ▶ {tname}{_args_preview}")
                _tool_t0 = time.time()
                if _attachment_content_ready and tname in {
                    "file_processor", "file_controller", "open_app",
                    "screen_process", "screen_locate",
                }:
                    result = (
                        "The attachment has already been extracted or rendered into the conversation. "
                        "Do not open it or capture the screen; answer the user's question using that context now."
                    )
                    self.ui.write_log("SYS: Attachment already available - skipped redundant tool")
                else:
                    result = self._execute_tool(tname, targs)
                # Completion status with duration: the missing "real-time"
                # feedback for tool calls.
                _dt = time.time() - _tool_t0
                _ok = "failed" not in (result or "").lower() and "error" not in (result or "").lower()
                _mark = "✓" if _ok else "✗"
                self.ui.write_log(f"SYS: {_mark} {tname} ({_dt:.1f}s)")

                # ── Auto-capture tool state for continuity ──
                try:
                    if self._cur_session_id:
                        if tname == "computer_control":
                            action = targs.get("action", "")
                            if action in ("type", "smart_type", "click", "hotkey", "press"):
                                set_tool_memory("browser", self._cur_session_id, "last_action", action)
                        elif tname == "run_terminal":
                            cmd = targs.get("command", "")
                            cwd = targs.get("cwd", "")
                            set_tool_memory("terminal", self._cur_session_id, "last_command", cmd[:200])
                            if cwd:
                                set_tool_memory("terminal", self._cur_session_id, "last_cwd", cwd)
                            if result and "error" not in result.lower():
                                set_tool_memory("terminal", self._cur_session_id, "last_success", "true")
                        elif tname == "screen_process":
                            text = targs.get("text", "")
                            set_tool_memory("vision", self._cur_session_id, "last_question", text[:200])
                        elif tname == "file_controller":
                            fpath = targs.get("path", "") or targs.get("destination", "") or ""
                            if fpath:
                                set_tool_memory("file_controller", self._cur_session_id, "last_path", fpath)
                        elif tname == "code_helper":
                            fp = targs.get("file_path", "") or targs.get("output_path", "") or ""
                            if fp:
                                set_tool_memory("code_helper", self._cur_session_id, "last_file", fp)
                except Exception:
                    pass

                if result != "__SILENT__":
                    all_silent = False
                    _tool_results.append((tname, result))

                _img_match = re.match(r'^__IMG__:(.+?):__IMG__(?:\|(.*))?$', str(result), re.DOTALL) if result != "__SILENT__" else None
                if _img_match:
                    _has_image = True
                    raw_b64 = _img_match.group(1)
                    img_text = (_img_match.group(2) or "").strip()
                    images_list = raw_b64.split("|")
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "user":
                            messages[i] = {**messages[i], "images": images_list}
                            break
                    tool_msg = {
                        "role": "tool",
                        "content": img_text or "Image captured.",
                    }
                else:
                    tool_msg: dict = {
                        "role":    "tool",
                        "content": "Done." if result == "__SILENT__" else str(result),
                    }
                if tc_id is not None:
                    tool_msg["tool_call_id"] = tc_id

                messages.append(tool_msg)
                self._conversation.append(tool_msg)

            if all_silent:
                _saved_name: str | None = None
                for _tc in final_tool_calls:
                    _fn = _tc.get("function", {})
                    if _fn.get("name") == "save_memory":
                        _a = _fn.get("arguments", {})
                        if isinstance(_a, str):
                            try:
                                _a = json.loads(_a)
                            except Exception:
                                _a = {}
                        if isinstance(_a, dict) and _a.get("key") == "name" and _a.get("value"):
                            _saved_name = str(_a["value"])
                            break
                _ack = f"Got it, {_saved_name}." if _saved_name else "Noted."
                _amsg = {"role": "assistant", "content": _ack}
                messages.append(_amsg)
                self._conversation.append(_amsg)
                self._deliver_response(_ack, source)
                self._extract_and_log(_ack)
                self.speak(_ack)
                break

            if _round > 200:
                _amsg = {"role": "assistant", "content": "Task completed after maximum rounds."}
                messages.append(_amsg)
                self._conversation.append(_amsg)
                self._deliver_response(_amsg["content"], source)
                self._extract_and_log(_amsg["content"])
                self.speak(_amsg["content"])
                break

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        for msg in self._conversation:
            msg.pop("images", None)

        try:
            new_turns = self._conversation[self._conv_saved_idx:]
            if new_turns:
                save_conversation(new_turns, self._cur_session_id)
                self._conv_saved_idx = len(self._conversation)
                # Auto-title on first user message
                if len(self._conversation) <= 2:
                    session = get_session(self._cur_session_id)
                    if session and session.get("title") == "New Chat":
                        auto_title_session(self._cur_session_id)
                # Real-time session info (turns/tokens/updated_at) after save
                try:
                    self.ui._win._session_refresh_sig.emit()
                except Exception:
                    pass
        except Exception as e:
            print(f"[Memory] ⚠️ Conversation save error: {e}")

        # ── Turn-reflection (async via queue) ────────────────────────────
        try:
            turn_count = len(self._conversation) // 2
            if turn_count > 0 and turn_count % 10 == 0 and self._cur_session_id:
                recent_text = " ".join(
                    (m.get("content") or "")[:120] for m in self._conversation[-4:]
                )
                self._enqueue_reflection(recent_text, self._cur_session_id)
        except Exception as e:
            print(f"[Timeline] Reflection queue error: {e}")

        # ── Importance decay (every 20 turns) ────────────────────────────
        try:
            if turn_count > 0 and turn_count % 20 == 0:
                self._decay_old_memories()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # STT listening loops
    # ------------------------------------------------------------------

    def _listen_whisper(self) -> None:
        vad = _VADBuffer()
        barge = _BargeDetector() if (self._streaming_tts and self._config.get("tts_barge_in", True)) else None
        q: queue.Queue = queue.Queue(maxsize=200)

        def callback(indata, frames, time_info, status):
            if barge is None:
                with self._speaking_lock:
                    is_speaking = self._speaking
                if is_speaking or self.ui.muted:
                    return
            elif self.ui.muted:
                return
            try:
                q.put_nowait(indata.tobytes())
            except queue.Full:
                pass

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE_IN,
                channels=CHANNELS,
                dtype="int16",
                blocksize=BLOCK_SIZE,
                callback=callback,
            ):
                self.ui.write_log("SYS: Mic active (Whisper STT).")
                while True:
                    try:
                        data = q.get(timeout=0.1)
                        chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        if barge is not None:
                            with self._speaking_lock:
                                is_speaking = self._speaking
                            if is_speaking:
                                rms = float(np.sqrt(np.mean(chunk ** 2)))
                                if barge.speech(rms):
                                    self.stop_speaking()
                                    self._signal_cancel()
                                    self.ui.set_state("THINKING")
                                    audio = vad.process(chunk)
                                    if audio is not None:
                                        text = self._stt.transcribe(audio)
                                        if text.strip():
                                            self._process_message(text)
                                continue
                        audio = vad.process(chunk)
                        if audio is not None:
                            self.stop_speaking()
                            self._signal_cancel()
                            self.ui.set_state("THINKING")
                            text = self._stt.transcribe(audio)
                            if text.strip():
                                self._process_message(text)
                    except queue.Empty:
                        pass
        except Exception as e:
            print(f"[STT-Whisper] Mic error: {e}")
            traceback.print_exc()

    def _listen_vosk(self) -> None:
        barge = _BargeDetector() if (self._streaming_tts and self._config.get("tts_barge_in", True)) else None
        q: queue.Queue = queue.Queue(maxsize=200)

        def callback(indata, frames, time_info, status):
            if barge is None:
                with self._speaking_lock:
                    is_speaking = self._speaking
                if is_speaking or self.ui.muted:
                    return
            elif self.ui.muted:
                return
            try:
                q.put_nowait(indata.tobytes())
            except queue.Full:
                pass

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE_IN,
                channels=CHANNELS,
                dtype="int16",
                blocksize=4096,
                callback=callback,
            ):
                self.ui.write_log("SYS: Mic active (Vosk STT).")
                while True:
                    try:
                        data = q.get(timeout=0.1)
                        chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                        if barge is not None:
                            with self._speaking_lock:
                                is_speaking = self._speaking
                            if is_speaking:
                                rms = float(np.sqrt(np.mean(chunk ** 2)))
                                if barge.speech(rms):
                                    self.stop_speaking()
                                    self._signal_cancel()
                                    self.ui.set_state("THINKING")
                                    text, is_final = self._stt.process_chunk(data)
                                    if is_final and text.strip():
                                        self._process_message(text)
                                continue
                        text, is_final = self._stt.process_chunk(data)
                        if is_final and text.strip():
                            self.stop_speaking()
                            self._signal_cancel()
                            self._process_message(text)
                    except queue.Empty:
                        pass
        except Exception as e:
            print(f"[STT-Vosk] Mic error: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Text command loop (UI input box)
    # ------------------------------------------------------------------

    def _text_command_loop(self) -> None:
        while True:
            try:
                item = self._text_queue.get(timeout=0.5)
                if isinstance(item, tuple) and item[0] == "file_msg":
                    _, text, extra = item
                    print(f"[Orthos] file_msg received: text={text[:50]}... extra_keys={list(extra.keys())}")
                    try:
                        self._process_message(text, source="ui", extra_meta=extra)
                    except Exception as e:
                        print(f"[Orthos] ERROR in _process_message (file_msg): {e}")
                        import traceback
                        traceback.print_exc()
                elif isinstance(item, tuple) and item[0] == "action_msg":
                    # F7: Regenerate/Continue — handled on this worker thread
                    # so conversation mutation stays race-free.
                    self._run_chat_action(item[1])
                elif isinstance(item, str) and item.strip():
                    self._process_message(item)
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[Orthos] ERROR in _text_command_loop: {e}")
                import traceback
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self.ui.on_reconfigure = self.reconfigure

            # ── Metrics server (non-blocking) ──────────────────────────────
            try:
                start_metrics_server(9090)
            except Exception:
                pass

            # ── OpenTelemetry tracing ──────────────────────────────────────
            init_tracing()

            # ── ChromaDB health check ──────────────────────────────────────
            if self._chroma:
                try:
                    health = self._chroma.health()
                    self.ui.write_log(f"SYS: ChromaDB {health.get('status', 'unknown')}"
                                      f" — {health.get('count', 0)} entries")
                except Exception as e:
                    self.ui.write_log(f"SYS: ChromaDB init skipped — {e}")

            # ── LLM health ────────────────────────────────────────────────
            from core.llm_client import ensure_ollama_running, warmup_model, get_llm_provider
            _llm_prov = get_llm_provider()
            self.ui.write_log(f"SYS: Checking {_llm_prov}…")
            if ensure_ollama_running():
                self.ui.write_log(f"SYS: {_llm_prov} OK.")
            else:
                self.ui.write_log(f"ERR: {_llm_prov} unavailable.")

            # ── Config ────────────────────────────────────────────────────
            stt_engine   = self._config.get("stt_engine",   "whisper").lower()
            stt_language = self._config.get("stt_language", "auto")
            stt_model    = self._config.get("stt_model",    "base")
            tts_engine   = self._config.get("tts_engine",   "edgetts").lower()

            # ── Startup progress panel ────────────────────────────────────
            self.ui.show_startup_panel()

            _warmup_done = threading.Event()
            _stt_done    = threading.Event()

            # ── LLM warmup thread ─────────────────────────────────────────
            def _do_warmup():
                try:
                    static_prompt = _load_system_prompt()
                    warmup_model(system_prompt=static_prompt)
                    self.ui.write_log("SYS: LLM ready.")
                    self.ui.mark_startup_ready("llm")
                except Exception as e:
                    self.ui.write_log(f"ERR: LLM warmup — {e}")
                    self.ui.mark_startup_ready("llm", error=True)
                finally:
                    _warmup_done.set()

            # ── STT load thread ───────────────────────────────────────────
            def _do_stt():
                try:
                    self.ui.write_log(f"SYS: Loading {stt_engine.upper()} STT…")
                    if stt_engine == "vosk":
                        from core.stt import VoskSTT
                        self._stt = VoskSTT(
                            self._config.get("vosk_model_path"),
                            language=stt_language,
                        )
                    elif stt_engine == "gemini":
                        from core.stt import GeminiSTT
                        gemini_key = self._config.get("gemini_voice_api_key", "").strip()
                        if not gemini_key:
                            raise RuntimeError(
                                "Gemini STT requires 'gemini_voice_api_key' — "
                                "set it in Configure → Text-to-Speech → Voice API Key."
                            )
                        self._stt = GeminiSTT(gemini_key, language=stt_language)
                    elif stt_engine == "groq":
                        from core.stt import GroqSTT
                        groq_key = self._config.get("groq_api_key", "")
                        self._stt = GroqSTT(groq_key, language=stt_language)
                    else:
                        from core.stt import WhisperSTT
                        self._stt = WhisperSTT(stt_model, language=stt_language)
                    self.ui.write_log("SYS: STT ready.")
                    self.ui.mark_startup_ready("stt")
                except Exception as e:
                    self.ui.write_log(f"ERR: STT — {e}")
                    self.ui.mark_startup_ready("stt", error=True)
                finally:
                    _stt_done.set()

            # ── TTS load thread — does NOT block going online ─────────────
            def _do_tts():
                try:
                    self.ui.write_log(f"SYS: Loading {tts_engine.upper()} TTS…")
                    if tts_engine == "kokoro":
                        self.ui.write_log("SYS: Kokoro — loading model + compiling JIT…")
                        _os.environ.pop("HF_HUB_OFFLINE", None)
                        _os.environ.pop("TRANSFORMERS_OFFLINE", None)
                        _os.environ.pop("HF_DATASETS_OFFLINE", None)
                    from core.tts import create_tts_player
                    self._tts = create_tts_player(self._config)
                    self._streaming_tts = bool(getattr(self._tts, "supports_streaming", False))
                    if self._streaming_tts:
                        from core.llm_client import set_voice_mode
                        set_voice_mode(True)
                        try:
                            self._tts.warmup()
                        except Exception as e:
                            print(f"[TTS] warmup error: {e}")
                    self._tts_ready.set()
                    self.ui.write_log("SYS: TTS ready.")
                    self.ui.mark_startup_ready("tts")
                    self.ui.set_startup_status("● All systems ready.")
                    self.ui.hide_startup_panel()
                    self.speak("Orthos fully online.")
                except Exception as e:
                    import traceback as _tb; _tb.print_exc()
                    self.ui.write_log(f"ERR: TTS — {e}")
                    self.ui.mark_startup_ready("tts", error=True)
                    self._tts_ready.set()

            # ── Gemini Live persistent mode: wire session callbacks ──────────
            self._live_provider = None
            try:
                if _get_provider_config()["provider"] == "gemini_live":
                    from core.llm_provider import get_registry
                    _live_cfg = _get_provider_config()
                    _live = get_registry().get_provider(
                        "gemini_live",
                        api_key=_live_cfg.get("api_key", ""),
                        model=_live_cfg.get("model", ""),
                    )
                    _live.register_tool_handler(self._execute_tool)
                    _live.register_output_cb(self._on_live_output)
                    _live.register_user_cb(self._on_live_user)
                    _live.register_muted_getter(lambda: self.ui.muted)
                    _live.register_speaking_cb(self._on_live_speaking)
                    self._live_provider = _live
                    self.ui.write_log("SYS: Gemini Live persistent session mode.")
            except Exception as e:
                print(f"[Live] Callback wiring failed: {e}")

            # Launch all startup threads simultaneously
            self.ui.write_log("SYS: Loading systems in parallel…")
            threading.Thread(target=_do_warmup, daemon=True).start()
            if self._live_provider is not None:
                self._stt = None
                _stt_done.set()
                self.ui.write_log("SYS: STT skipped (Gemini Live handles mic input natively).")
            else:
                threading.Thread(target=_do_stt,    daemon=True).start()
            threading.Thread(target=_do_tts,    daemon=True).start()

            # ── Wait ONLY for STT + LLM (fast) ────────────────────────────
            _warmup_done.wait(timeout=60)
            _stt_done.wait(timeout=60)

            # ── Initialise session system ──────────────────────────────────
            self._init_session_system()
            self.ui.write_log(f"SYS: Session {self._cur_session_id[:13]}... ready.")

            # ── Go online immediately ──────────────────────────────────────
            self.ui.write_log("SYS: Orthos online.")
            self.ui.set_state("LISTENING")
            self.ui.set_startup_status("● Orthos online · Voice loading in background…")

            if not self._is_headless:
                threading.Thread(target=self._tts_worker,        daemon=True).start()
            threading.Thread(target=self._text_command_loop,  daemon=True).start()

            # ── Gateway (Telegram, etc.) ─────────────────────────────────────
            try:
                from gateway import Gateway
                self._gateway = Gateway(
                    process_callback=self._process_message,
                    log_callback=self.ui.write_log,
                    cancel_callback=self._signal_cancel,
                )
                self._gateway.start()
                self.register_response_callback(self._gateway.route_response)
                from gateway.log_capture import LogCapture
                self._log_capture = LogCapture(self._gateway.send_log)
                self._log_capture.start()
                from core.logging_setup import rehook_console_sink
                rehook_console_sink()
                self.ui.write_log("SYS: Terminal log capture started.")
            except Exception as e:
                self.ui.write_log(f"ERR: Gateway init — {e}")

            # ── MiniLM inline load (use boot-time memory service if available) ──
            self._memory_svc_ok = False
            try:
                from memory.memory_client import health as check_memory_svc
                svc = check_memory_svc()
                if svc and svc.get("status") == "ok":
                    self._memory_svc_ok = True
                    self.ui.write_log("SYS: Using preloaded memory service (boot-time).")
                else:
                    raise ConnectionError("Memory service not reachable")
            except Exception:
                try:
                    self.ui.write_log("SYS: Loading MiniLM L6 v2…")
                    from memory.vector_index import preload_model
                    preload_model()
                    self.ui.write_log("SYS: MiniLM ready.")
                    self.ui.write_log("SYS: Loading spaCy NER…")
                    from memory.entity_store import preload_ner
                    preload_ner()
                    self.ui.write_log("SYS: spaCy NER ready.")
                except Exception as e:
                    self.ui.write_log(f"ERR: local model load — {e}")

            # ── Background memory dreaming (background fact extraction) ──────
            try:
                from memory.dreaming import start_dream_loop
                start_dream_loop(interval_seconds=120)
                self.ui.write_log("SYS: Memory dreaming started.")
            except Exception as e:
                self.ui.write_log(f"ERR: Memory dreaming — {e}")

            # ── MCP Server connections ────────────────────────────────────
            try:
                from core.mcp_manager import get_global_manager, attach_runtime
                attach_runtime(
                    registry=_tool_registry,
                    tool_index=_tool_index,
                    loaded_set=_tool_active_full_schemas,
                )
                self._mcp_manager = get_global_manager()
                mcp_decls = self._mcp_manager.discover_and_register(_tool_registry)
                if mcp_decls:
                    self.ui.write_log(f"SYS: MCP — {len(mcp_decls)} tool(s) registered (hidden from payload, searchable via search_tools)")
                    print(f"[MCP] {len(mcp_decls)} MCP tool(s) registered")
            except Exception as e:
                self.ui.write_log(f"ERR: MCP init — {e}")
                self._mcp_manager = None

            # ── Warm vector + BM25 indexes (skip if memory service already loaded them) ──
            if not self._memory_svc_ok:
                try:
                    from memory.memory_manager import load_memory, get_recent_turns_for_vector
                    from memory.vector_index import rebuild_index
                    turns = get_recent_turns_for_vector(100)
                    facts = load_memory()
                    rebuild_index(turns, facts)
                    self.ui.write_log("SYS: Vector index built.")
                except Exception as e:
                    self.ui.write_log(f"ERR: Vector index — {e}")
                try:
                    from memory.bm25_search import rebuild_bm25_index
                    rebuild_bm25_index()
                    self.ui.write_log("SYS: BM25 index built.")
                except Exception as e:
                    self.ui.write_log(f"ERR: BM25 index — {e}")
            else:
                self.ui.write_log("SYS: Indexes already loaded via memory service.")

            # ── Block — STT loop or headless sleep ──────────────────────────
            if self._is_headless:
                self.ui.write_log("SYS: Headless mode active — Telegram bot running.")
                self.ui.write_log("SYS: Press Ctrl+C to stop.")
                while True:
                    import time as _time
                    _time.sleep(1)
            elif self._live_provider is not None:
                # Gemini Live: the persistent session streams the mic and
                # plays Gemini's native voice — no STT loop / TTS speaking.
                self.ui.write_log("SYS: Gemini Live — hands-free voice mode.")
                self.ui.set_state("LISTENING")
                while True:
                    import time as _time
                    _time.sleep(1)
            elif stt_engine == "vosk":
                self._listen_vosk()
            else:
                self._listen_whisper()

        except Exception as e:
            self.ui.write_log(f"ERR: Init failed — {e}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    print("[Orthos] Starting with improved modules: Loguru · SecretsManager · "
          "ToolRegistry · LLM Provider · Prometheus · ChromaDB · OpenTelemetry · HealthChecks")

    # ── Pre-import torch in background immediately ─────────────────────────
    def _preload_torch():
        try:
            import torch  # noqa: F401
        except Exception:
            pass
    threading.Thread(target=_preload_torch, daemon=True).start()
    # ───────────────────────────────────────────────────────────────────────

    if _ARGS.headless:
        from core.headless_ui import HeadlessUI
        ui = HeadlessUI()
    else:
        ui = OrthosUI("face.png")

    def runner():
        if not _ARGS.headless:
            ui.wait_for_api_key()

        ui.write_log("SYS: Checking dependencies…")
        cfg = _load_config()
        _install_done = threading.Event()

        def _do_install():
            try:
                from core.installer import install_for_config
                install_for_config(cfg, log=ui.write_log)
            except Exception as e:
                ui.write_log(f"ERR: Dependency install — {e}")
            finally:
                _install_done.set()

        threading.Thread(target=_do_install, daemon=True).start()
        _install_done.wait()

        orthos = Orthos(ui)
        try:
            orthos.run()
        except KeyboardInterrupt:
            print("\n[Orthos] Shutting down…")
        finally:
            try:
                orthos._close_current_session()
            except Exception:
                pass

    threading.Thread(target=runner, daemon=True).start()
    if _ARGS.headless:
        try:
            while True:
                import time as _time
                _time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Orthos] Shutting down…")
    else:
        ui.root.mainloop()


if __name__ == "__main__":
    main()
