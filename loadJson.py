"""
 Loads JSON data into a dynamic SQLAlchemy schema in PostgreSQL.
 Supports both descriptive and measurement data with flexible column mapping.

 Version: 1.1.8.5
 Date: 2026-03-22
 Changes:
   - Fiexed GUI not launching on some Linux distros due to missing Tkinter modules; added robust import handling and fallback.
   - Fixed `AttributeError: module 'uuid' has no attribute 'uuid7'` for Python < 3.11 by adding a fallback to the `uuid6` library. Requires `pip install uuid6`.
   - Corrected UUID handling: Uses standard `uuid` library. `uuid.UUID()` is used for parsing any existing UUID version from JSON/CLI, and `uuid.uuid7()` is used for generating new UUIDs. Removed `uuid6` dependency.
   - Added status area to GUI to show which file is being processed.
   - Added pre-flight check to detect and report duplicate SAMPLE_IDs in JSON, then abort.
   - Corrected measurement_id precedence: JSON file > GUI/CLI override > New UUID.
   - Changed new UUID generation from v4 to v7.
   - Corrected logic to ensure measurement_id from JSON is used when no CLI/GUI override is provided.
   - Fixed dry-run output to reflect the correct final measurement_id.
   - Cleaned up function signatures by removing unused parameters.
   - Removed redundant/conflicting code blocks for dry-run, logging, and measurement_id generation.
   - DB section: row1 = URI/Name/Port, row2 = Schema/Username/Password; fixed Show/Hide password button (uses *).
   - Combined Log level + Log file in one row; halved log level width; checkboxes in 2x2 grid; remember window size.
   - Fixed logging call that caused 'TypeError: not all arguments converted during string formatting'.
   - Switched DB section to a macOS-friendly two-column ttk grid: Database URI on top spanning both columns; added show/hide for password.
   - Added robust logging to (source)/Logs with rotating file handler.
   - Default log filename now includes date: loadJson_YYYY-MM-DD.log.
   - Added --log-level and --log-file CLI flags.
   - Relative --log-file paths resolve against the script folder; on Windows, paths
     starting with '\' or '\\' are treated as absolute.
   - Redacts passwords in DB URLs within logs.
   - Logs split to stdout (INFO/WARN) and stderr (ERROR/CRITICAL).
   - Tkinter GUI buttons now enforce colors on macOS; fallback to light gray with black text if themes override colors.
   - If MEASUREMENT_ID exists in descriptive_data, generate a new UUID.
   - Apply this new ID to both descriptive_data and all measurement_data entries.
   - Emits a warning when this happens.
    - Added --no-db-check to skip DB connectivity check in dry run mode.
    - Added --dry-run to validate JSON and DB connection without performing inserts.
    - Added --verbose and --debug flags for more detailed logging.  
"""

import json
import uuid
import logging
import argparse
import sys
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
from logging.handlers import RotatingFileHandler

# --- UUIDv7 Backport for Python < 3.11 ---
# Requires `pip install uuid6`
try:
    from uuid import uuid7
except ImportError:
    try:
        from uuid6 import uuid7
    except ImportError:
        # Fallback to uuid4 if uuid6 is not installed
        print("Warning: 'uuid6' library not found. Falling back to uuid.uuid4() for new UUIDs. "
              "For time-sortable UUIDs (v7), please run: pip install uuid6", file=sys.stderr)
        uuid7 = uuid.uuid4


# Optional GUI imports (Tkinter)
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from tkinter import font as tkfont
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False
    tkfont = None

# Absolute folder where this script resides
SCRIPT_DIR = Path(__file__).resolve().parent

# GUI preferences file
PREFS_DIR = SCRIPT_DIR / "Config"
PREFS_PATH = PREFS_DIR / "loadJson_gui_prefs.json"

def load_prefs() -> dict:
    try:
        if PREFS_PATH.exists():
            return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def save_prefs(data: dict) -> None:
    try:
        PREFS_DIR.mkdir(parents=True, exist_ok=True)
        PREFS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as e:
        logging.getLogger().warning(f"Could not save GUI prefs: {e}")

# GUI font (bold) helper
def _get_bold_label_font():
    try:
        f = tkfont.nametofont("TkDefaultFont").copy()
        f.configure(weight="bold")
        return f
    except Exception:
        return ("Segoe UI", 10, "bold")

# GUI font (bold) helper for buttons
def _get_bold_button_font():
    try:
        f = tkfont.nametofont("TkDefaultFont").copy()
        f.configure(weight="bold")
        return f
    except Exception:
        return ("Segoe UI", 10, "bold")


def _extract_measurement_id_from_json(desc: dict, measurements: list) -> uuid.UUID | None:
    """
    Try to find a measurement_id in the JSON payload.
    Accepts 'MEASUREMENT_ID' or 'measurement_id' in desc or rows.
    Ensures all discovered IDs (if multiple) are identical; otherwise raises.
    Returns uuid.UUID or None if not present.
    """
    candidates = set()

    # From descriptive section
    for key in ("MEASUREMENT_ID", "measurement_id"):
        val = (desc or {}).get(key)
        if val:
            candidates.add(str(val).strip())

    # From measurement rows
    for row in (measurements or []):
        for key in ("MEASUREMENT_ID", "measurement_id"):
            val = row.get(key)
            if val:
                candidates.add(str(val).strip())

    if not candidates:
        return None

    if len(candidates) > 1:
        raise ValueError(f"Conflicting MEASUREMENT_ID values in JSON: {sorted(candidates)}")

    only = next(iter(candidates))
    try:
        return uuid.UUID(only)
    except Exception as e:
        raise ValueError(f"Invalid MEASUREMENT_ID format in JSON: {only}") from e


from sqlalchemy import (
    Boolean, create_engine, Column, Integer, Float, Text, DateTime,
    ForeignKey, text
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import func


# ----------------------------
# Config / Logging defaults
# ----------------------------
DEFAULT_SCHEMA = "color_data"
DB_USERNAME = "pgadmin"
DB_PASSWORD = "postgres"
DB_PREFIX = "postgresql+psycopg2://"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_URI = "10.211.55.9"

def _ensure_logs_dir() -> Path:
    """Create (source folder)/Logs and return its path."""
    logs_dir = SCRIPT_DIR / "Logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir

def _redact_url(url: str) -> str:
    """
    Redact passwords in DB URLs like postgresql://user:pass@host:port/db?...
    """
    if not url:
        return url
    return re.sub(r"(://[^:]+:)([^@]+)(@)", r"\1****\3", url)

def setup_logging(app_name: str = "loadJson", log_level: str = "INFO", log_file_override: Path | None = None) -> logging.Logger:
    """
    Configure root logger:
    - File: (source)/Logs/{app_name}_{YYYY-MM-DD}.log (rotating, 5MB x 7)
    - Console: INFO+ to stdout, ERROR+ to stderr
    - Uncaught exceptions are logged at CRITICAL
    """
    logger = logging.getLogger()
    if getattr(logger, "_configured", False):
        # Re-configure level if called again
        level = getattr(logging, log_level.upper(), logging.INFO)
        for handler in logger.handlers:
            handler.setLevel(level)
        logger.setLevel(level)
        return logger

    # Levels
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    # Clear existing handlers from any previous basicConfig
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logs_dir = _ensure_logs_dir()
    if log_file_override:
        log_file = Path(log_file_override)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        log_file = logs_dir / f"{app_name}_{datetime.now().strftime('%Y-%m-%d')}.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler (rotating)
    fh = RotatingFileHandler(str(log_file), maxBytes=5 * 1024 * 1024, backupCount=7, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console: INFO+ to stdout
    class StdoutFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.levelno < logging.ERROR

    ch_info = logging.StreamHandler(stream=sys.stdout)
    ch_info.setLevel(level)
    ch_info.addFilter(StdoutFilter())
    ch_info.setFormatter(formatter)
    logger.addHandler(ch_info)

    # Console: ERROR+ to stderr
    ch_err = logging.StreamHandler(stream=sys.stderr)
    ch_err.setLevel(logging.ERROR)
    ch_err.setFormatter(formatter)
    logger.addHandler(ch_err)

    # Uncaught exceptions → logger.critical with traceback
    def _excepthook(exc_type, exc_value, exc_tb):
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_tb))
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    logger._configured = True
    logger.debug(f"Logging initialized. File: {log_file}")
    return logger

# This will be re-assigned in main after args are parsed
logger = logging.getLogger(__name__)

def log(msg: str, level: int = logging.INFO):
    # Use the global logger instance
    logger.log(level, msg)


# ----------------------------
# Schema utilities
# ----------------------------
def ensure_schema(engine, schema_name: str):
    """Ensure the target schema exists; create if missing."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = :s"),
            {"s": schema_name}
        )
        if result.fetchone():
            log(f"Schema '{schema_name}' exists.", logging.INFO)
        else:
            log(f"Schema '{schema_name}' does not exist. Creating...", logging.WARNING)
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema_name}"'))
            conn.commit()
            log(f"Schema '{schema_name}' created.", logging.INFO)


# ----------------------------
# Model factory (dynamic schema)
# ----------------------------
def make_models(schema: str):
    Base = declarative_base()

    class DescriptiveData(Base):
        __tablename__ = "descriptive_data"
        __table_args__ = {"schema": schema}

        measurement_id = Column(PG_UUID(as_uuid=True), default=uuid7, primary_key=True)
        project = Column(Text)
        template = Column(Text)
        measurement = Column(Text)
        print_date = Column(DateTime(timezone=True), default=func.now())
        creation_date = Column(DateTime(timezone=True), default=func.now())
        target_id = Column(Text)    
        originator = Column(Text)   
        device_model = Column(Text)
        device_serial_number = Column(Text)
        sample_backing = Column(Text)   
        measurement_illumination = Column(Text) 
        measurement_angle = Column(Text) 
        measurement_condition = Column(Text)    
        measurement_filter = Column(Text)   
        number_of_fields = Column(Integer)
        number_of_sets = Column(Integer)
        row_length = Column(Integer)
        data_format = Column(Text)  
        sample_id = Column(Integer)
        sample_name = Column(Text)
        cmyk = Column(Boolean) 
        rgb = Column(Boolean)
        xyz = Column(Boolean)
        lab = Column(Boolean)
        spectral = Column(Boolean)
        description = Column(Text) 
        status = Column(Text) 
        firstdate = Column(DateTime(timezone=True), default=func.now())
        lastdate = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
        version = Column(Integer) 

        def __repr__(self):
            return f"<DescriptiveData(measurement_id={self.measurement_id}, originator={self.originator}, ...)>"

    class MeasurementData(Base):
        __tablename__ = "measurement_data"
        __table_args__ = {"schema": schema}

        measurement_id = Column(PG_UUID(as_uuid=True), ForeignKey(f'{schema}.descriptive_data.measurement_id'), primary_key=True)
        sample_id = Column(Integer, primary_key=True)

        cmyk_c = Column(Float)
        cmyk_m = Column(Float)
        cmyk_y = Column(Float)
        cmyk_k = Column(Float)
        xyz_x = Column(Float)
        xyz_y = Column(Float)
        xyz_z = Column(Float)
        lab_l = Column(Float)
        lab_a = Column(Float)
        lab_b = Column(Float)

        # Spectral columns (380–780nm step 10)
        spectral_380 = Column(Float)
        spectral_390 = Column(Float)
        spectral_400 = Column(Float)
        spectral_410 = Column(Float)
        spectral_420 = Column(Float)
        spectral_430 = Column(Float)
        spectral_440 = Column(Float)
        spectral_450 = Column(Float)
        spectral_460 = Column(Float)
        spectral_470 = Column(Float)
        spectral_480 = Column(Float)
        spectral_490 = Column(Float)
        spectral_500 = Column(Float)
        spectral_510 = Column(Float)
        spectral_520 = Column(Float)
        spectral_530 = Column(Float)
        spectral_540 = Column(Float)
        spectral_550 = Column(Float)
        spectral_560 = Column(Float)
        spectral_570 = Column(Float)
        spectral_580 = Column(Float)
        spectral_590 = Column(Float)
        spectral_600 = Column(Float)
        spectral_610 = Column(Float)
        spectral_620 = Column(Float)
        spectral_630 = Column(Float)
        spectral_640 = Column(Float)
        spectral_650 = Column(Float)
        spectral_660 = Column(Float)
        spectral_670 = Column(Float)
        spectral_680 = Column(Float)
        spectral_690 = Column(Float)
        spectral_700 = Column(Float)
        spectral_710 = Column(Float)
        spectral_720 = Column(Float)
        spectral_730 = Column(Float)
        spectral_740 = Column(Float)
        spectral_750 = Column(Float)
        spectral_760 = Column(Float)
        spectral_770 = Column(Float)
        spectral_780 = Column(Float)

        def __repr__(self):
            return f"<MeasurementData(measurement_id={self.measurement_id}, sample_id={self.sample_id}, ...)>"

    return Base, DescriptiveData, MeasurementData


# ----------------------------
# Mapping helper
# ----------------------------
def build_descriptive_kwargs(desc_json: dict, model_cls, force_guid: uuid.UUID | None = None):
    """
    Build kwargs for SQLAlchemy model from JSON:
    - Only include keys that exist as columns on the model.
    - Ignore model fields missing from JSON.
    - Ignore extra JSON keys not in the model.
    - Light type coercion for Integer/DateTime.
    - JSON keys expected UPPER_SNAKE_CASE; model columns lower_snake_case.
    """
    cols = {c.name: c.type for c in model_cls.__table__.columns}
    out = {}
    for k, v in (desc_json or {}).items():
        col = k.lower()
        if col not in cols:
            continue
        if v is None or v == "":
            continue
        t = cols[col]
        try:
            tname = type(t).__name__.lower()
            if "integer" in tname:
                out[col] = int(v)
            elif "datetime" in tname:
                # Accept ISO ("YYYY-MM-DD" or with time)
                try:
                    out[col] = datetime.fromisoformat(v)
                except Exception:
                    # Try simple date-only
                    try:
                        out[col] = datetime.strptime(v, "%Y-%m-%d")
                    except Exception:
                        out[col] = v
            else:
                out[col] = v
        except Exception:
            out[col] = v
    if "measurement_id" in cols and force_guid:
        out["measurement_id"] = force_guid
    return out


# ----------------------------
# GUI (optional Tkinter)
# ----------------------------
def run_gui_with_prefs(initial_args) -> argparse.Namespace | None:
    """
    Launch a Tkinter GUI to capture all arguments.
    Returns an argparse.Namespace with the selected values, or None if cancelled.
    """
    if not TK_AVAILABLE:
        raise RuntimeError("Tkinter is not available in this environment.")

    prefs = load_prefs()

    root = tk.Tk()
    root.title("loadJson — Arguments")
    style = ttk.Style(root)
    style.configure("TLabelframe.Label", font=("TkDefaultFont", 11, "bold"))

    geom = prefs.get("window_geometry", "900x415+5+10")
    try:
        root.geometry(geom)
    except Exception:
        root.geometry("900x415+5+10")
   
    mainfrm = ttk.Frame(root, padding=16)
    mainfrm.pack(fill="both", expand=True)

    def field(row, label, var, width=60, browse=None, wrap=False, wrap_px=220):
        label_text = label.rstrip(":") + ":"
        lbl_kwargs = {"text": label_text, "anchor": "e"}
        if wrap:
            lbl_kwargs["wraplength"] = wrap_px
            lbl_kwargs["justify"] = "left"
        lbl = ttk.Label(mainfrm, **lbl_kwargs)
        try:
            lbl.configure(font=_get_bold_label_font())
        except Exception: pass
        lbl.grid(row=row, column=0, sticky="e", pady=4)

        ent = ttk.Entry(mainfrm, textvariable=var, width=width)
        ent.grid(row=row, column=1, sticky="we", pady=4)
        if browse == "dir":
            ttk.Button(mainfrm, text="Browse…", command=lambda: var.set(filedialog.askdirectory(initialdir=var.get() or str(SCRIPT_DIR)))).grid(row=row, column=2, padx=6)
        elif browse == "file":
            ttk.Button(mainfrm, text="Browse…", command=lambda: var.set(filedialog.askopenfilename(initialdir=str((Path(var.get()).parent if var.get() else SCRIPT_DIR)), filetypes=[('JSON','*.json'), ('All','*.*')]))).grid(row=row, column=2, padx=6)
        elif browse == "savefile":
            ttk.Button(mainfrm, text="Browse…", command=lambda: var.set(filedialog.asksaveasfilename(initialdir=str(SCRIPT_DIR), defaultextension=".log"))).grid(row=row, column=2, padx=6)
        return ent

    def build_db_connection_section(parent, v_db_uri, v_db_name, v_db_port, v_username, v_password, v_schema):
        frame = ttk.LabelFrame(parent, text="Database Connection", padding=(12, 10))
        for c in (1, 3, 5):
            frame.columnconfigure(c, weight=1)
        frame.columnconfigure(6, weight=0)

        def add_pair(row, col0, label_text, textvar, show=None):
            label_txt = label_text.rstrip(":") + ":"
            lbl = ttk.Label(frame, text=label_txt, anchor="e")
            try:
                lbl.configure(font=_get_bold_label_font())
            except Exception: pass
            lbl.grid(row=row, column=col0, sticky="e", padx=(0, 8), pady=4)
            ent = ttk.Entry(frame, textvariable=textvar, show=show)
            ent.grid(row=row, column=col0 + 1, sticky="ew", padx=(0, 0), pady=4)
            return lbl, ent

        add_pair(0, 0, "Database URI",  v_db_uri)
        add_pair(0, 2, "Database name", v_db_name)
        add_pair(0, 4, "Database port", v_db_port)
        add_pair(1, 0, "Schema",   v_schema)
        add_pair(1, 2, "Username", v_username)
        _, ent_pass = add_pair(1, 4, "Password", v_password, show="*")
        
        show_pw_var = tk.BooleanVar(value=False)
        def toggle_pw():
            ent_pass.configure(show="" if show_pw_var.get() else "*")
        
        btn_pw = ttk.Checkbutton(frame, text="Show", variable=show_pw_var, command=toggle_pw)
        btn_pw.grid(row=1, column=6, sticky="w", padx=(6, 0), pady=4)
        return frame

    v_path = tk.StringVar(value=prefs.get("path", initial_args.path if hasattr(initial_args, "path") else str(SCRIPT_DIR / "Data")))
    v_json = tk.StringVar(value=prefs.get("json_file", initial_args.json_file if hasattr(initial_args, "json_file") else "printer.cie.json"))
    v_guid = tk.StringVar(value=prefs.get("guid", getattr(initial_args, "guid", "") or ""))
    v_db = tk.StringVar(value=prefs.get("db_uri", getattr(initial_args, "db_uri", "") or ""))
    v_db_port = tk.StringVar(value=prefs.get("db_port", getattr(initial_args, "db_port", "") or ""))
    v_db_name = tk.StringVar(value=prefs.get("db_name", getattr(initial_args, "db_name", "") or ""))
    v_username = tk.StringVar(value=prefs.get("username", getattr(initial_args, "username", "") or ""))
    v_password = tk.StringVar(value=prefs.get("password", getattr(initial_args, "password", "") or ""))
    v_schema = tk.StringVar(value=prefs.get("schema", getattr(initial_args, "schema", DEFAULT_SCHEMA)))
    v_loglevel = tk.StringVar(value=prefs.get("log_level", getattr(initial_args, "log_level", "INFO")))
    v_logfile = tk.StringVar(value=prefs.get("log_file", getattr(initial_args, "log_file", "") or ""))
    v_debug = tk.BooleanVar(value=prefs.get("debug", getattr(initial_args, "debug", False)))
    v_verbose = tk.BooleanVar(value=prefs.get("verbose", getattr(initial_args, "verbose", False)))
    v_dry = tk.BooleanVar(value=prefs.get("dry_run", getattr(initial_args, "dry_run", False)))
    v_nocheck = tk.BooleanVar(value=prefs.get("no_db_check", getattr(initial_args, "no_db_check", False)))

    mainfrm.columnconfigure(1, weight=1)
    row = 0
    field(row, "Path (folder with JSON file)", v_path, browse="dir"); row += 1
    field(row, "JSON file name", v_json, browse="file"); row += 1
    field(row, "Measurement ID (UUID, optional)", v_guid, wrap=True); row += 1
    
    db_section = build_db_connection_section(mainfrm, v_db, v_db_name, v_db_port, v_username, v_password, v_schema)
    db_section.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=8, pady=6); row += 1

    logrow = ttk.Frame(mainfrm)
    logrow.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
    logrow.columnconfigure(1, weight=1)
    logrow.columnconfigure(3, weight=1)
    
    _lbl = ttk.Label(logrow, text="Log level:", anchor="e")
    try: _lbl.configure(font=_get_bold_label_font())
    except Exception: pass
    _lbl.grid(row=0, column=0, sticky="e", padx=(0,8))
    ttk.Combobox(logrow, textvariable=v_loglevel, values=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], state="readonly", width=10).grid(row=0, column=1, sticky="w")
    
    _lf = ttk.Label(logrow, text="Log file (optional override):", anchor="e")
    try: _lf.configure(font=_get_bold_label_font())
    except Exception: pass
    _lf.grid(row=0, column=2, sticky="e", padx=(12,8))
    ttk.Entry(logrow, textvariable=v_logfile).grid(row=0, column=3, sticky="ew")
    ttk.Button(logrow, text="Browse…", command=lambda: v_logfile.set(filedialog.asksaveasfilename(title="Select log file", initialdir=str(SCRIPT_DIR), defaultextension=".log"))).grid(row=0, column=4, padx=6)
    row += 1

    chkfrm = ttk.Frame(mainfrm)
    chkfrm.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
    chkfrm.columnconfigure(0, weight=1)
    chkfrm.columnconfigure(1, weight=1)
    ttk.Checkbutton(chkfrm, text="Dry run (no inserts)", variable=v_dry).grid(row=0, column=0, sticky="w", padx=4, pady=2)
    ttk.Checkbutton(chkfrm, text="Skip DB connectivity check in dry run", variable=v_nocheck).grid(row=0, column=1, sticky="w", padx=4, pady=2)
    ttk.Checkbutton(chkfrm, text="Debug", variable=v_debug).grid(row=1, column=0, sticky="w", padx=4, pady=2)
    ttk.Checkbutton(chkfrm, text="Verbose", variable=v_verbose).grid(row=1, column=1, sticky="w", padx=4, pady=2)
    row += 1

    ttk.Separator(mainfrm).grid(row=row, column=0, columnspan=3, sticky="we", pady=8); row += 1

    def make_colored_button(parent, text, command, variant="primary"):
        fg, bg, afg, abg = ("black", "#1abc9c", "black", "#16a085")
        if variant == "danger":
            fg, bg, afg, abg = ("black", "#DC143C", "black", "#B22222")
        btn = tk.Button(parent, text=text, command=command, fg=fg, bg=bg, activeforeground=afg, activebackground=abg, relief="raised", bd=1, highlightthickness=0, padx=12, pady=6, font=_get_bold_button_font())
        return btn

    msg_var = tk.StringVar(value="")
    ttk.Label(mainfrm, textvariable=msg_var, foreground="#B22222").grid(row=row, column=0, columnspan=3, sticky="ew", pady=(6,0)); row += 1

    btnfrm = ttk.Frame(mainfrm)
    btnfrm.grid(row=row, column=0, columnspan=3, sticky="e", pady=8); row += 1

    # --- Status Area ---
    status_frame = ttk.LabelFrame(mainfrm, text="Status", padding=(12, 10))
    status_frame.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=8, pady=6)
    status_frame.grid_remove() # Hide by default
    status_var = tk.StringVar()
    status_label = ttk.Label(status_frame, textvariable=status_var, wraplength=800, justify="left")
    status_label.pack(fill="x", expand=True)
    row += 1


    def on_cancel():
        prefs_to_save = {"window_geometry": root.winfo_geometry()}
        save_prefs(prefs_to_save)
        root.destroy()
        sys.exit(0)

    def on_execute():
        # --- Validation ---
        msg_var.set("") # Clear previous errors
        _json_path = Path(v_path.get().strip()) / v_json.get().strip()
        if not _json_path.exists():
            msg_var.set(f"Error: JSON file not found: {_json_path}"); return

        try:
            if v_guid.get().strip():
                uuid.UUID(v_guid.get().strip())
        except ValueError:
            msg_var.set(f"Error: Invalid Measurement ID format. Must be a valid UUID."); return

        # --- GUI Feedback ---
        status_var.set(f"Processing file: {_json_path.resolve()}")
        status_frame.grid() # Show status frame
        btn_exec.config(state="disabled", text="Processing...")
        btn_cancel.config(state="disabled")
        root.update_idletasks() # Force GUI to redraw immediately

        # --- Save Prefs & Quit ---
        payload = {
            "path": v_path.get(), "json_file": v_json.get(), "guid": v_guid.get(),
            "db_uri": v_db.get(), "db_port": v_db_port.get(), "db_name": v_db_name.get(),
            "username": v_username.get(), "password": v_password.get(), "schema": v_schema.get(),
            "log_level": v_loglevel.get(), "log_file": v_logfile.get(),
            "debug": bool(v_debug.get()), "verbose": bool(v_verbose.get()),
            "dry_run": bool(v_dry.get()), "no_db_check": bool(v_nocheck.get()),
            "window_geometry": root.winfo_geometry(),
        }
        save_prefs(payload)
        
        # Short delay to ensure user sees the message, then close
        root.after(1500, root.quit)


    btn_cancel = make_colored_button(btnfrm, "Cancel", on_cancel, variant="danger")
    btn_exec = make_colored_button(btnfrm, "Execute", on_execute, variant="primary")
    btn_cancel.pack(side="left", padx=6)
    btn_exec.pack(side="left", padx=6)
    root.protocol("WM_DELETE_WINDOW", on_cancel)

    root.mainloop()
    root.destroy()

    return argparse.Namespace(
        path=v_path.get(), json_file=v_json.get(), guid=v_guid.get().strip() or None,
        db_uri=v_db.get().strip() or None, db_port=v_db_port.get().strip() or None,
        db_name=v_db_name.get().strip() or None, username=v_username.get().strip() or None,
        password=v_password.get().strip() or None, schema=v_schema.get().strip() or DEFAULT_SCHEMA,
        log_level=v_loglevel.get(), log_file=v_logfile.get().strip() or None,
        debug=bool(v_debug.get()), verbose=bool(v_verbose.get()),
        dry_run=bool(v_dry.get()), no_db_check=bool(v_nocheck.get()), gui=True,
    )


# ----------------------------
# Core
# ----------------------------
def process_file(json_file: Path, db_uri: str, dry_run: bool = False, no_db_check: bool = False, schema: str = DEFAULT_SCHEMA, force_guid: uuid.UUID | None = None):
    engine = create_engine(db_uri, echo=False)
    log("SQLAlchemy engine created.", logging.DEBUG)

    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        log(f"Error: JSON file not found at '{json_file}'", logging.CRITICAL)
        raise SystemExit(1)
    except json.JSONDecodeError:
        log(f"Error: Could not decode JSON from '{json_file}'. Check for syntax errors.", logging.CRITICAL)
        raise SystemExit(1)

    desc = data.get("descriptive_data", {}) or {}
    measurements = data.get("measurement_data", []) or []

    # Determine final measurement_id: JSON > CLI/GUI > New UUIDv7
    json_guid = _extract_measurement_id_from_json(desc, measurements)
    guid_final = json_guid or force_guid or uuid7()
    log(f"Using measurement_id: {guid_final}", logging.INFO)
    if json_guid:
        log("ID source: Found in JSON file.", logging.DEBUG)
    elif force_guid:
        log("ID source: Provided by CLI/GUI override.", logging.DEBUG)
    else:
        log("ID source: Generated new UUIDv7.", logging.DEBUG)

    # --- DRY-RUN LOGIC ---
    if dry_run:
        log('Dry-run mode: Skipping all database inserts.', logging.INFO)
        log(f"Dry-run against DB: {_redact_url(db_uri)}", logging.INFO)
        
        Base, DescriptiveData, MeasurementData = make_models(schema)
        desc_kwargs = build_descriptive_kwargs(desc, DescriptiveData, force_guid=guid_final)
        
        log("\n--- DRY RUN SAMPLE OUTPUT ---")
        log(f"Schema: {schema}")
        log("DescriptiveData (kwargs that will be inserted):")
        log(json.dumps(desc_kwargs, indent=4, default=str))
        
        if measurements:
            log("\nFirst 3 MeasurementData rows (after normalization):")
            for i, row in enumerate(measurements[:3]):
                row["MEASUREMENT_ID"] = guid_final
                log(json.dumps(row, indent=4, default=str))
        
        if not no_db_check:
            try:
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                log(f"DB connectivity check: OK -> {_redact_url(str(engine.url))}")
            except Exception as e:
                log(f"DB connectivity check FAILED: {e}", logging.ERROR)
        
        log("--- END DRY RUN ---")
        return # Exit after dry run

    # --- LIVE RUN LOGIC ---
    if not no_db_check:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log(f"DB connectivity: OK -> {_redact_url(str(engine.url))}", logging.INFO)
        except Exception as e:
            log(f"DB connectivity FAILED -> {_redact_url(str(engine.url))}\n{e}", logging.CRITICAL)
            raise SystemExit(2)

    ensure_schema(engine, schema)
    Base, DescriptiveData, MeasurementData = make_models(schema)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    with Session() as session:
        try:
            # Check for existing measurement_id
            existing = session.query(DescriptiveData).filter_by(measurement_id=guid_final).first()
            if existing:
                log(f"Measurement ID {guid_final} already exists. Aborting.", logging.ERROR)
                raise SystemExit(3)

            # --- PRE-FLIGHT CHECK FOR DUPLICATE SAMPLE_IDs ---
            if measurements:
                sample_ids = [row.get("SAMPLE_ID") for row in measurements]
                id_counts = Counter(sample_ids)
                duplicates = {sid: count for sid, count in id_counts.items() if count > 1 and sid is not None}
                if duplicates:
                    log(f"Error: Duplicate SAMPLE_ID values found in JSON file. Cannot insert.", logging.CRITICAL)
                    for sid, count in duplicates.items():
                        log(f"  - SAMPLE_ID '{sid}' appears {count} times.", logging.ERROR)
                    raise SystemExit(4)

            # Prepare descriptive data
            descriptive_entry = DescriptiveData(**build_descriptive_kwargs(desc, DescriptiveData, force_guid=guid_final))
            session.add(descriptive_entry)

            # Prepare measurement data
            cleaned_measurements = []
            for row in measurements:
                insert_data = {"measurement_id": guid_final, "sample_id": int(row["SAMPLE_ID"])}
                for key, value in row.items():
                    col_name = key.lower()
                    # Map common data types
                    if col_name in ("cmyk_c", "cmyk_m", "cmyk_y", "cmyk_k", "xyz_x", "xyz_y", "xyz_z", "lab_l", "lab_a", "lab_b") or col_name.startswith("spectral_"):
                        insert_data[col_name] = float(value) if value is not None else None
                cleaned_measurements.append(insert_data)

            # Bulk insert for performance
            if cleaned_measurements:
                session.bulk_insert_mappings(MeasurementData, cleaned_measurements)
            
            session.commit()
            log("Database transaction committed successfully.", logging.INFO)

        except (SQLAlchemyError, ValueError) as e:
            log(f"An error occurred during the database operation: {e}", logging.CRITICAL)
            logger.exception("Database operation failed")
            session.rollback()
            raise SystemExit(1)
        except Exception as e:
            log(f"An unexpected error occurred: {e}", logging.CRITICAL)
            logger.exception("Unexpected error")
            session.rollback()
            raise SystemExit(1)
        finally:
            engine.dispose()


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Load measurement JSON into Postgres.")
    parser.add_argument("--path", default=str(SCRIPT_DIR / "Data"), help="Folder path containing JSON file")
    parser.add_argument("--json_file", default="printer.cie.json", help="JSON file name")
    parser.add_argument("--guid", help="Override MEASUREMENT_ID (UUID). Overrides JSON value if JSON has no ID.")
    parser.add_argument("--db-uri", help="Override database connection host/IP")
    parser.add_argument("--db-port", help="Override database port")
    parser.add_argument("--db-name", help="Override database name")
    parser.add_argument("--username", help=f"Database username (default: {DB_USERNAME})")
    parser.add_argument("--password", help=f"Database password (default: {DB_PASSWORD})")
    parser.add_argument("--dry-run", action="store_true", help="Validate and log but do not insert into database")
    parser.add_argument("--no-db-check", action="store_true", help="Skip DB connectivity check during --dry-run")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, help=f"Target schema name (default: {DEFAULT_SCHEMA})")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging (overrides --log-level).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose (INFO) logging.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Log verbosity (default: INFO)")
    parser.add_argument("--log-file", help="Optional override path for log file.")
    parser.add_argument("--gui", action="store_true", help="Launch Tkinter GUI to enter arguments")
    args = parser.parse_args()

    # If GUI requested (and available), collect args via GUI
    if args.gui:
        if not TK_AVAILABLE:
            logging.critical("GUI requested but Tkinter is not available.")
            sys.exit(1)
        gui_args = run_gui_with_prefs(args)
        if gui_args is None:
            print("Operation cancelled by user.")
            sys.exit(0)
        args = gui_args

    # Determine desired log level
    log_level = "DEBUG" if args.debug else ("INFO" if args.verbose else args.log_level)
    
    log_override = None
    if args.log_file:
        raw = Path(args.log_file).expanduser()
        log_override = (SCRIPT_DIR / raw).resolve() if not raw.is_absolute() else raw

    # Setup logging using the parsed arguments
    global logger
    logger = setup_logging(app_name="loadJson", log_level=log_level, log_file_override=log_override)
    
    log("Starting loadJson.py", logging.INFO)

    json_file = Path(args.path) / args.json_file
    if not json_file.exists():
        log(f"JSON file not found: {json_file}", logging.CRITICAL)
        raise SystemExit(1)

    # Determine measurement_id from --guid flag
    force_guid = None
    if args.guid:
        try:
            force_guid = uuid.UUID(args.guid)
        except ValueError:
            log(f"Invalid --guid format: {args.guid}", logging.CRITICAL)
            raise SystemExit(1)

    # Build DB connection URI from parts
    db_user = args.username or DB_USERNAME
    db_pass = args.password or DB_PASSWORD
    db_host = args.db_uri or DB_URI
    db_port = args.db_port or DB_PORT
    db_name = args.db_name or DB_NAME
    db_uri_full = f"{DB_PREFIX}{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    log(f"Using DB URI: {_redact_url(db_uri_full)}", logging.INFO)
    log(f"Schema: {args.schema}", logging.INFO)
    log(f"Processing file: {json_file}", logging.INFO)
    log(f"CLI --guid override: {force_guid or 'Not provided'}", logging.INFO)

    try:
        process_file(
            json_file=json_file,
            db_uri=db_uri_full,
            dry_run=args.dry_run,
            no_db_check=args.no_db_check,
            schema=args.schema,
            force_guid=force_guid,
        )
        if not args.dry_run:
            log("✅ All measurement records inserted successfully.", logging.INFO)
    except SystemExit as e:
        log(f"Process halted with exit code {e.code}.", logging.WARNING if e.code == 0 else logging.ERROR)
    except Exception as e:
        log(f"An unhandled error occurred in process_file: {e}", logging.CRITICAL)
        logger.exception("Unhandled error in main process")


if __name__ == "__main__":
    main()
