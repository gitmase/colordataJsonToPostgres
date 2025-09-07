"""
 Loads JSON data into a dynamic SQLAlchemy schema in PostgreSQL.
 Supports both descriptive and measurement data with flexible column mapping.

 Version: 1.1.8
 Date: 2025-09-07
 Changes:
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
- Date: 2025-09-07
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
        logger.error("Exception occurred", exc_info=True)
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
        logger.error("Exception occurred", exc_info=True)
        raise ValueError(f"Invalid MEASUREMENT_ID format in JSON: {only}") from e


from sqlalchemy import (
    Boolean, create_engine, Column, Integer, Float, Text, DateTime,
    ForeignKey, text
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import func




# === ENHANCED LOGGING SETUP WITH QUIET MODE AND LOGFILE OVERRIDE ===
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

log_folder = os.path.join(os.path.dirname(__file__), "Logs")
os.makedirs(log_folder, exist_ok=True)

# Placeholder for dynamic log_file (set after parsing args)
log_file = None

# Setup basic logger and temporarily attach console output
temp_console_handler = logging.StreamHandler()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[temp_console_handler]
)
logger = logging.getLogger(__name__)
# === END ENHANCED LOGGING SETUP WITH QUIET MODE AND LOGFILE OVERRIDE ===


# ----------------------------
# Config / Logging defaults
# ----------------------------
DEFAULT_SCHEMA = "color_data"  # "color_measurement"
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
        return logger

    # Levels
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(logging.DEBUG)  # capture all; handlers filter

    logs_dir = _ensure_logs_dir()
    if log_file_override:
        log_file = Path(log_file_override)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Use dated default log filename (keeps override working)
        log_file = logs_dir / f"{app_name}_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    # else:
    #    log_file = logs_dir / f"{app_name}.log"

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

    logger._configured = True  # type: ignore[attr-defined]
    logger.debug(f"Logging initialized. File: {log_file}")
    return logger

def log(msg: str, level: int = logging.INFO):
    logging.log(level, msg)


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

        measurement_id = Column(PG_UUID(as_uuid=True), default=uuid.uuid4, primary_key=True)
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
def build_descriptive_kwargs(desc_json: dict, model_cls, force_guid: None):
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
def run_gui_with_prefs(initial_args) -> argparse.Namespace:
    """
    Launch a Tkinter GUI to capture all arguments.
    Returns an argparse.Namespace with the selected values.
    """
    if not TK_AVAILABLE:
        raise RuntimeError("Tkinter is not available in this environment.")

    prefs = load_prefs()

    root = tk.Tk()
    root.title("loadJson — Arguments")
    style = ttk.Style(root)
    style.configure("TLabelframe.Label", font=("TkDefaultFont", 11, "bold"))  # affects all

    # Restore previous window size/position if available
    geom = prefs.get("window_geometry")
    if geom:
        try:
            root.geometry(geom)
        except Exception:
            root.geometry("900x415+5+10")
    else:
        root.geometry("900x415+5+10")

    # root.geometry("900x415+5+10")
   
    mainfrm = ttk.Frame(root, padding=16)
    mainfrm.pack(fill="both", expand=True)

    # Helpers
    def field(row, label, var, width=60, browse=None, wrap=False, wrap_px=220):
        # Ensure colon suffix
        base = label.rstrip()
        label_text = base if base.endswith(":") else base + ":"

        lbl_kwargs = {"text": label_text, "anchor": "e"}  # right-align within cell
        if wrap:
            lbl_kwargs["wraplength"] = wrap_px
            lbl_kwargs["justify"] = "left"  # text lines themselves left-justified
        # Bold label font
        lbl = ttk.Label(mainfrm, **lbl_kwargs)
        try:
            lbl.configure(font=_get_bold_label_font())
        except Exception:
            pass
        # Right-justify in grid cell
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

    def boolbox(row, label, var):
        cb = ttk.Checkbutton(mainfrm, text=label, variable=var)
        cb.grid(row=row, column=1, sticky="w", pady=4)
        return cb

    def build_db_connection_section(parent,
                                    v_db_uri,
                                    v_db_name,
                                    v_db_port,
                                    v_username,
                                    v_password,
                                    v_schema):
        """
        Database section in two rows, three fields per row.

        Row 0: Database URI | Database name | Database port
        Row 1: Schema       | Username      | Password [+ Show/Hide]

        Returns (frame, refs).
        """
        frame = ttk.LabelFrame(parent, text="Database Connection", padding=(12, 10))

        # Grid columns inside frame:
        # 0: lbl1, 1: ent1, 2: lbl2, 3: ent2, 4: lbl3, 5: ent3, 6: extras (e.g., show/hide)
        for c in (1, 3, 5):
            frame.columnconfigure(c, weight=1)  # entries expand
        frame.columnconfigure(6, weight=0)

        def add_pair(row, col0, label_text, textvar, show=None):
            base = label_text.rstrip()
            label_txt = base if base.endswith(":") else base + ":"
            lbl = ttk.Label(frame, text=label_txt, anchor="e")
            try:
                lbl.configure(font=_get_bold_label_font())
            except Exception:
                pass
            lbl.grid(row=row, column=col0, sticky="e", padx=(0, 8), pady=4)
            ent = ttk.Entry(frame, textvariable=textvar, show=show)
            ent.grid(row=row, column=col0 + 1, sticky="ew", padx=(0, 0), pady=4)
            return lbl, ent

        # Row 0
        _, ent_uri   = add_pair(0, 0, "Database URI",  v_db_uri)
        _, ent_dbn   = add_pair(0, 2, "Database name", v_db_name)
        _, ent_port  = add_pair(0, 4, "Database port", v_db_port)

        # Row 1
        _, ent_schema = add_pair(1, 0, "Schema",   v_schema)
        _, ent_user   = add_pair(1, 2, "Username", v_username)

        # Password with Show/Hide
        mask_char = "*"
        _, ent_pass = add_pair(1, 4, "Password", v_password, show=mask_char)
        show_pw = tk.BooleanVar(value=False)

         # Show/Hide password toggle
        show_pw = tk.BooleanVar(value=False)
        def toggle_pw():
            ent_pass.configure(show="" if show_pw.get() else "•")
            btn_pw.configure(text="Hide" if show_pw.get() else "Show")
            show_pw.set(not show_pw.get())
            ent_pass.focus_set()
            ent_pass.selection_range(0, tk.END)
         
        btn_pw = ttk.Button(frame, text="Show", command=toggle_pw, width=6)
        # Place to the right of password entry
        btn_pw.grid(row=1, column=6, sticky="w", padx=(6, 0), pady=4)
        
        # Start with focus on the URI field
        ent_uri.focus_set()

        return frame, {
            "ent_uri": ent_uri,
            "ent_dbname": ent_dbn,
            "ent_port": ent_port,
            "ent_schema": ent_schema,
            "ent_user": ent_user,
            "ent_pass": ent_pass,
            "btn_pw": btn_pw,
            "show_pw": show_pw,
        }
    v_path = tk.StringVar(value=prefs.get("path", initial_args.path if hasattr(initial_args, "path") else str(SCRIPT_DIR / "Data")))
    v_json = tk.StringVar(value=prefs.get("json_file", initial_args.json_file if hasattr(initial_args, "json_file") else "printer.cie.json"))
    v_measid = tk.StringVar(value=prefs.get("measurement_id", getattr(initial_args, "measurement_id", "") or ""))
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

    # Grid config
    for c in range(3):
        mainfrm.columnconfigure(c, weight=1)

    row = 0
    field(row, "Path (folder with JSON file)", v_path, browse="dir"); row += 1
    field(row, "JSON file name", v_json, browse="file"); row += 1
    field(row, "Measurement ID (UUID, optional)", v_measid, wrap=True); row += 1
    # Two-column Database Connection section (macOS-friendly)
    db_section, _dbrefs = build_db_connection_section(
        mainfrm, v_db, v_db_name, v_db_port, v_username, v_password, v_schema
    )
    # Span across all three main columns
    db_section.grid(row=row, column=0, columnspan=3, sticky="nsew", padx=8, pady=6)
    row += 1

    # Combined row: Log level (left) + Log file (right)
    logrow = ttk.Frame(mainfrm)
    logrow.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
    for c in (1, 3):
        logrow.columnconfigure(c, weight=1)  # expand combo (col1) and file entry (col3)

    # Log level (left side)
    _lbl = ttk.Label(logrow, text="Log level:", anchor="e")
    try: _lbl.configure(font=_get_bold_label_font())
    except Exception: pass
    _lbl.grid(row=0, column=0, sticky="e", padx=(0,8))
    cmb = ttk.Combobox(logrow, textvariable=v_loglevel, values=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], state="readonly", width=10)
    cmb.grid(row=0, column=1, sticky="w")

    # Log file (right side)
    _lf = ttk.Label(logrow, text="Log file (optional override):", anchor="e")
    try: _lf.configure(font=_get_bold_label_font())
    except Exception: pass
    _lf.grid(row=0, column=2, sticky="e", padx=(12,8))
    ent_lf = ttk.Entry(logrow, textvariable=v_logfile)
    ent_lf.grid(row=0, column=3, sticky="ew")
    ttk.Button(logrow, text="Browse…", command=lambda: v_logfile.set(filedialog.asksaveasfilename(title="Select log file", initialdir=str(SCRIPT_DIR), defaultextension=".log"))).grid(row=0, column=4, padx=6)
    row += 1

    # Options checkboxes in 2 rows x 2 columns
    chkfrm = ttk.Frame(mainfrm)
    chkfrm.grid(row=row, column=0, columnspan=3, sticky="ew", pady=4)
    chkfrm.columnconfigure(0, weight=1)
    chkfrm.columnconfigure(1, weight=1)
    ttk.Checkbutton(chkfrm, text="Dry run (no inserts)", variable=v_dry).grid(row=0, column=0, sticky="w", padx=4, pady=2)
    ttk.Checkbutton(chkfrm, text="Skip DB connectivity check in dry run", variable=v_nocheck).grid(row=0, column=1, sticky="w", padx=4, pady=2)
    ttk.Checkbutton(chkfrm, text="Debug", variable=v_debug).grid(row=1, column=0, sticky="w", padx=4, pady=2)
    ttk.Checkbutton(chkfrm, text="Verbose", variable=v_verbose).grid(row=1, column=1, sticky="w", padx=4, pady=2)
    row += 1

    sep = ttk.Separator(mainfrm); sep.grid(row=row, column=0, columnspan=3, sticky="we", pady=8); row += 1

    # Buttons (Cancel red, Execute blue-green)
    # Helper: reliable colored buttons across platforms
    def make_colored_button(parent, text, command, variant="primary"):
        """
        Use tk.Button to ensure bg/fg colors apply on macOS/Windows/Linux.
        Variants:
          - "danger": red
          - "primary": blue-green (teal)
        Falls back to light gray with black text if the platform/theme refuses custom colors.
        """
        # Desired colors by variant (text always black)
        if variant == "danger":
            fg, bg, afg, abg = ("black", "#DC143C", "black", "#B22222")
        else:
            fg, bg, afg, abg = ("black", "#1abc9c", "black", "#16a085")  # teal primary

        # Construct button
        btn = tk.Button(
            parent, text=text, command=command,
            fg=fg, bg=bg, activeforeground=afg, activebackground=abg,
            relief="raised", bd=1, highlightthickness=0, padx=12, pady=6,
            font=_get_bold_button_font(),
        )

        # Try to make colors stick on macOS Aqua
        try:
            ws = btn.tk.call("tk", "windowingsystem")
            if ws == "aqua":
                # Help Aqua respect custom colors (colored outline)
                btn.configure(highlightbackground=bg)

            # If theme still overrides, detect and fallback to neutral
            applied_bg = str(btn.cget("bg")).lower()
            if applied_bg.startswith("system") or applied_bg in ("", "white"):
                # fallback: neutral light gray with black text
                btn.configure(fg="black", bg="#D9D9D9", activeforeground="black", activebackground="#C8C8C8")
        except Exception:
            # Any error -> safe fallback
            btn.configure(fg="black", bg="#D9D9D9", activeforeground="black", activebackground="#C8C8C8")

        return btn

    msg_var = tk.StringVar(value="")
    msgfrm = ttk.Frame(mainfrm)
    msgfrm.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(6,0))
    msglbl = ttk.Label(msgfrm, textvariable=msg_var, foreground="#B22222")
    msglbl.grid(row=0, column=0, sticky="w")
    row += 1

    btnfrm = ttk.Frame(mainfrm)
    btnfrm.grid(row=row, column=0, columnspan=3, sticky="e", pady=8)

    def on_cancel():
        # Save window geometry before exit
        try:
            gp = load_prefs()
        except Exception:
            gp = {}
        gp["window_geometry"] = root.winfo_geometry()
        try:
            save_prefs(gp)
        except Exception:
            pass
        root.destroy()
        raise SystemExit(0)
    def on_execute():
        # persist prefs
        payload = {
            "path": v_path.get(),
            "json_file": v_json.get(),
            "measurement_id": v_measid.get(),
            "db_uri": v_db.get(),
            "db_port": v_db_port.get(),
            "db_name": v_db_name.get(),
            "username": v_username.get(),
            "password": v_password.get(),
            "schema": v_schema.get(),
            "log_level": v_loglevel.get(),
            "log_file": v_logfile.get(),
            "debug": bool(v_debug.get()),
            "verbose": bool(v_verbose.get()),
            "dry_run": bool(v_dry.get()),
            "no_db_check": bool(v_nocheck.get()),
            "window_geometry": root.winfo_geometry(),
        }
        # --- Duplicate measurement_id guard in GUI before closing ---
        try:
            # Resolve schema and DB URI
            _username = (v_username.get().strip() or DB_USERNAME)
            _password = (v_password.get().strip() or DB_PASSWORD)
            _host = (v_db.get().strip() or DB_URI)
            _port = (v_db_port.get().strip() or DB_PORT)
            _dbname = (v_db_name.get().strip() or DB_NAME)
            _schema = (v_schema.get().strip() or DEFAULT_SCHEMA)
            _db_uri_full = f"{DB_PREFIX}{_username}:{_password}@{_host}:{_port}/{_dbname}"

            # Determine target measurement_id (GUI override if provided)
            _force_guid = None
            _meid_txt = (v_measid.get() or "").strip()
            if _meid_txt:
                try:
                    _force_guid = uuid.UUID(_meid_txt)
                except Exception as _e:
                    msg_var.set(f"Error: Invalid measurement_id format: {_meid_txt}")
                    return

            # Load JSON and extract ID if needed
            import os, json as _json
            _json_path = os.path.join(v_path.get().strip(), v_json.get().strip())
            try:
                with open(_json_path, "r", encoding="utf-8") as _f:
                    _data = _json.load(_f)
            except Exception as _e:
                msg_var.set(f"Error: Could not open JSON file: {_json_path}")
                return

            _desc = _data.get("descriptive_data", {}) or {}
            _meas = _data.get("measurement_data", []) or []
            _json_guid = None
            try:
                _json_guid = _extract_measurement_id_from_json(_desc, _meas)
            except Exception as _e:
                msg_var.set(f"Error: {_e}")
                return
            _guid_final = _force_guid or _json_guid or uuid.uuid4()

            # Build engine once
            _engine = create_engine(_db_uri_full, echo=False)

            # If user did NOT check 'Skip DB connectivity check', preflight ping
            if not bool(v_nocheck.get()):
                try:
                    with _engine.connect() as _conn:
                        _conn.execute(text("SELECT 1"))
                except Exception as _e:
                    logger.error("Exception occurred", exc_info=True)
                    logger.critical(f"DB connectivity FAILED -> {_redact_url(_db_uri_full)} | {_e}")
                    try:
                        _engine.dispose()
                    except Exception:
                        pass
                    msg_var.set("Error: DB connectivity failed; insert halted")
                    return

            # Query DB for existing descriptive_data with same measurement_id
            try:
                with _engine.connect() as _conn:
                    _res = _conn.execute(text(f'SELECT 1 FROM "{_schema}".descriptive_data WHERE measurement_id = :mid LIMIT 1'), {"mid": _guid_final})
                    if _res.first():
                        msg_var.set(f"Error: Measurement with the measurement_id: {_guid_final} exist; insert halted")
                        try:
                            _engine.dispose()
                        except Exception:
                            pass
                        return
            except Exception as _e:
                # Surface DB errors in the GUI
                msg_var.set(f"Error: DB check failed — {_e}")
                try:
                    _engine.dispose()
                except Exception:
                    pass
                return
            try:
                _engine.dispose()
            except Exception:
                pass
        except Exception as _e:
            msg_var.set(f"Error: {_e}")
            return
        save_prefs(payload)
        root.quit()

    # Use helper for reliable colored buttons
    btn_cancel = make_colored_button(btnfrm, "Cancel", on_cancel, variant="danger")
    btn_exec = make_colored_button(btnfrm, "Execute", on_execute, variant="primary")
    btn_cancel.pack(side="left", padx=6)
    btn_exec.pack(side="left", padx=6)
    root.protocol("WM_DELETE_WINDOW", on_cancel)

    root.mainloop()
    root.destroy()

    # Build argparse-like namespace
    ns = argparse.Namespace(
        path=v_path.get(),
        json_file=v_json.get(),
        measurement_id=v_measid.get().strip() or None,
        db_uri=v_db.get().strip() or None,
        db_port=v_db_port.get().strip() or None,
        db_name=v_db_name.get().strip() or None,    
        username=v_username.get().strip() or None,
        password=v_password.get().strip() or None,
        schema=v_schema.get().strip() or DEFAULT_SCHEMA,
        log_level=v_loglevel.get(),
        log_file=v_logfile.get().strip() or None,
        debug=bool(v_debug.get()),
        verbose=bool(v_verbose.get()),
        dry_run=bool(v_dry.get()),
        no_db_check=bool(v_nocheck.get()),
        gui=True,
    )
    return ns


# ----------------------------
# Core
# ----------------------------
def process_file(json_file: Path, guid: None, db_uri: str, dry_run: bool = False, no_db_check: bool = False, schema: str = DEFAULT_SCHEMA, force_guid: uuid.UUID | None = None, gui: bool = False):
    engine = create_engine(db_uri, echo=False)

    log("SQLAlchemy engine created.", logging.DEBUG)

    # ## PREFLIGHT CONNECTIVITY CHECK ##
    if not no_db_check:
        try:
            with engine.connect() as _conn:
                _conn.execute(text("SELECT 1"))
            log(f"DB connectivity: OK -> {_redact_url(str(engine.url))}", logging.INFO)
        except Exception as e:
            logger.error("Exception occurred", exc_info=True)
            log(f"DB connectivity FAILED -> {_redact_url(str(engine.url))}\\n{e}", logging.CRITICAL)
            try:
                engine.dispose()
            finally:
                raise SystemExit(2)


    # Ensure schema exists (auto-create)
    # --- EARLY DRY-RUN EXIT (no DB mutations) ---
    if dry_run:
        log('Dry-run mode: Skipping all database inserts.', logging.INFO)
        log(f"Dry-run against DB: {_redact_url(db_uri)}", logging.INFO)
        import json as _json
        with open(json_file, "r", encoding="utf-8") as _f:
            _data = json.load(_f)
        _desc = _data.get("descriptive_data", {}) or {}
        _measurements = _data.get("measurement_data", []) or []
        # Determine measurement_id
        _guid_final = force_guid or _extract_measurement_id_from_json(_desc, _measurements) or uuid.uuid4()
        # Normalize rows & basic validation
        _cleaned = []
        for _row in _measurements:
            _row["MEASUREMENT_ID"] = _guid_final
            if not _row.get("MEASUREMENT_ID"):
                raise ValueError(f"Missing MEASUREMENT_ID in row: {_row}")
            if "SAMPLE_ID" not in _row:
                raise ValueError(f"Missing SAMPLE_ID in row: {_row}")
            _cleaned.append(_row)
        # Duplicate guard
        from collections import Counter as _Counter
        _pairs = [(_r["MEASUREMENT_ID"], _r["SAMPLE_ID"]) for _r in _cleaned]
        _dupes = [k for k, v in _Counter(_pairs).items() if v > 1]
        if _dupes:
            raise ValueError(f"Duplicate (measurement_id, sample_id) pairs found: {_dupes}")
        # Build descriptive kwargs (force_guid applied inside)
        _Base, _DescriptiveData, _MeasurementData = make_models(schema)
        _desc_kwargs = build_descriptive_kwargs(_desc, _DescriptiveData, force_guid=force_guid)
        logger.info("\n--- DRY RUN SAMPLE OUTPUT ---")
        logger.info("Schema: %s", schema)
        logger.info("DescriptiveData (kwargs that will be inserted):")
        logger.info(_json.dumps(_desc_kwargs, indent=4, default=str))
        if _cleaned:
            logger.info("\nFirst 3 MeasurementData rows:")
            for _sample_row in _cleaned[:3]:
                logger.info(_json.dumps(_sample_row, indent=4, default=str))
        # Optional DB connectivity check
        if not no_db_check:
            try:
                with engine.connect() as _conn:
                    _conn.execute(text("SELECT 1"))
                logger.info(f"DB connectivity: OK -> {_redact_url(str(engine.url))}")
            except Exception as _e:
                logger.error("Exception occurred", exc_info=True)
                logger.critical(f"DB connectivity FAILED -> {_redact_url(str(engine.url))} | {str(_e)}")
                try:
                    engine.dispose()
                finally:
                    raise SystemExit(2)

        logger.info("--- END SAMPLE OUTPUT ---\n")
        logger.info("\nPlanned index statements:")
        logger.info(f'CREATE INDEX IF NOT EXISTS idx_measurement_sample ON "{schema}".measurement_data (measurement_id, sample_id)')
        logger.info(f'CREATE INDEX IF NOT EXISTS idx_measurement_id ON "{schema}".measurement_data (measurement_id)')
        logger.info(f'CREATE INDEX IF NOT EXISTS idx_sample_id ON "{schema}".measurement_data (sample_id)\n')
        return
    ensure_schema(engine, schema)

    # Build models bound to this schema
    Base, DescriptiveData, MeasurementData = make_models(schema)

    # Create tables if missing
    Base.metadata.create_all(engine)
    # Ensure new columns exist (idempotent for existing DBs)
    with engine.connect() as conn:
        conn.execute(text(f'ALTER TABLE "{schema}".descriptive_data ADD COLUMN IF NOT EXISTS project TEXT'))
        conn.execute(text(f'ALTER TABLE "{schema}".descriptive_data ADD COLUMN IF NOT EXISTS template TEXT'))
        conn.execute(text(f'ALTER TABLE "{schema}".descriptive_data ADD COLUMN IF NOT EXISTS parsed_date TEXT'))
        conn.commit()
    log("Schema up-to-date: ensured project, template, parsed_date columns.", logging.INFO)
    # Create indexes for performance
    with engine.connect() as conn:
        conn.execute(text(f'CREATE INDEX IF NOT EXISTS idx_measurement_sample ON "{schema}".measurement_data (measurement_id, sample_id)'))
        conn.execute(text(f'CREATE INDEX IF NOT EXISTS idx_measurement_id ON "{schema}".measurement_data (measurement_id)'))
        conn.execute(text(f'CREATE INDEX IF NOT EXISTS idx_sample_id ON "{schema}".measurement_data (sample_id)'))
        conn.commit()
    log("Indexes ensured on measurement_data table.", logging.INFO)

    Session = sessionmaker(bind=engine)
    session = Session()

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

        # === PATCH: Generate new MEASUREMENT_ID if does not exist ===
        descriptive_data = data.get("descriptive_data", {})
        if not ("MEASUREMENT_ID" in descriptive_data and descriptive_data["MEASUREMENT_ID"].strip()):
            logger.info("MEASUREMENT_ID missing in descriptive_data. Generating new MEASUREMENT_ID.")
            new_measurement_id = str(uuid.uuid4())
            descriptive_data["MEASUREMENT_ID"] = new_measurement_id
            for item in data.get("measurement_data", []):
                item["MEASUREMENT_ID"] = new_measurement_id
        # === END PATCH ===


        desc = data.get("descriptive_data", {}) or {}
        measurements = data.get("measurement_data", []) or []

        # Determine measurement_id: prefer CLI arg, else JSON, else generate new
        guid_final = force_guid or _extract_measurement_id_from_json(desc, measurements) or uuid.uuid4()

        log(f"Using measurement_id: {guid_final}", logging.INFO)
        
        # Force the SAME guid into each measurement row and basic validation
        cleaned_measurements = []
        for row in measurements:
            row["MEASUREMENT_ID"] = guid_final  # keep as uuid.UUID consistently
            if not row.get("MEASUREMENT_ID"):
                raise ValueError(f"Missing MEASUREMENT_ID in row: {row}")
            if "SAMPLE_ID" not in row:
                raise ValueError(f"Missing SAMPLE_ID in row: {row}")
            cleaned_measurements.append(row)

        # Duplicate (measurement_id, sample_id) guard
        pairs = [(r["MEASUREMENT_ID"], r["SAMPLE_ID"]) for r in cleaned_measurements]
        dupes = [k for k, v in Counter(pairs).items() if v > 1]
        if dupes:
            raise ValueError(f"Duplicate (measurement_id, sample_id) pairs found: {dupes}")

        # DescriptiveData: filtered kwargs (ignore missing + ignore extra) with SAME guid
        descriptive_entry = DescriptiveData(**build_descriptive_kwargs(desc, DescriptiveData, force_guid=force_guid))

        # Dry-run: print sample + optional DB connectivity check, then exit
        if dry_run:
            log('Dry-run mode: Skipping all database inserts.', logging.INFO)
            log(f"Dry-run against DB: {_redact_url(db_uri)}", logging.INFO)
            # Show sample data
            import json as _json
            logger.info("\n--- DRY RUN SAMPLE OUTPUT ---")
            logger.info("Schema: %s", schema)
            logger.info("DescriptiveData:")
            logger.info(_json.dumps(build_descriptive_kwargs(desc, DescriptiveData, force_guid=force_guid), indent=4, default=str))
            if cleaned_measurements:
                logger.info("\nFirst 3 MeasurementData rows:")
                for sample_row in cleaned_measurements[:3]:
                    logger.info(_json.dumps(sample_row, indent=4, default=str))
            # Check DB connectivity if not disabled
            if not no_db_check:
                try:
                    with engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    logger.info(f"DB connectivity: OK -> {_redact_url(str(engine.url))}")
                except Exception as e:
                    logger.error("Exception occurred", exc_info=True)
                    logger.info(f"DB connectivity: FAILED -> {_redact_url(str(engine.url))}\n{e}")
            
            logger.info("--- END SAMPLE OUTPUT ---\n")
            logger.info("\nPlanned index statements:")
            logger.info(f'CREATE INDEX IF NOT EXISTS idx_measurement_sample ON "{schema}".measurement_data (measurement_id, sample_id)')
            logger.info(f'CREATE INDEX IF NOT EXISTS idx_measurement_id ON "{schema}".measurement_data (measurement_id)')
            logger.info(f'CREATE INDEX IF NOT EXISTS idx_sample_id ON "{schema}".measurement_data (sample_id)\n')
        
            session.close()


        # --- Duplicate measurement_id guard (non-GUI/CLI path) ---
        try:
            with engine.connect() as _conn:
                _dup = _conn.execute(text(f'SELECT 1 FROM "{schema}".descriptive_data WHERE measurement_id = :mid LIMIT 1'), {"mid": guid_final}).first()
            if _dup:
                _msg = f"Measurement with the measurement_id: {guid_final} exist; insert halted"
                log(_msg, logging.CRITICAL)
                return
        except Exception as _e:
            logger.error("Exception occurred during duplicate check", exc_info=True)
            raise
            # --- Insert section ---
        try:
            session.add(descriptive_entry)
            session.commit()
            session.refresh(descriptive_entry)
            log("DescriptiveData entry committed successfully.", logging.INFO)
        except SQLAlchemyError as e:
            logger.error("Exception occurred", exc_info=True)
            session.rollback()
            log(f"Error inserting DescriptiveData: {e}", logging.ERROR)
            logging.getLogger().exception("DescriptiveData insert failed")
            raise

        for row in cleaned_measurements:
            base_fields = {
                "measurement_id": guid_final,  # enforce same UUID
                "sample_id": int(row["SAMPLE_ID"]),
                "cmyk_c": row.get("CMYK_C"),
                "cmyk_m": row.get("CMYK_M"),
                "cmyk_y": row.get("CMYK_Y"),
                "cmyk_k": row.get("CMYK_K"),
                "xyz_x": row.get("XYZ_X"),
                "xyz_y": row.get("XYZ_Y"),
                "xyz_z": row.get("XYZ_Z"),
                "lab_l": row.get("LAB_L"),
                "lab_a": row.get("LAB_A"),
                "lab_b": row.get("LAB_B"),
            }
            spectral_fields = {
                f'spectral_{wavelength}': row.get(f"SPECTRAL_{wavelength}", None)
                for wavelength in range(380, 781, 10)
            }
            measurement_entry = MeasurementData(**base_fields, **spectral_fields)
            try:
                session.add(measurement_entry)
                session.commit()
                session.refresh(measurement_entry)
                log(f"Measurement entry for sample_id {measurement_entry.sample_id} committed.", logging.INFO)
            except SQLAlchemyError as e:
                logger.error("Exception occurred", exc_info=True)
                session.rollback()
                log(f"Error inserting MeasurementData (sample_id={row.get('SAMPLE_ID')}): {e}", logging.ERROR)
                logging.getLogger().exception("MeasurementData insert failed")
                raise
            finally:
                session.close()


# ----------------------------
# CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Load measurement JSON into Postgres.")
    parser.add_argument("--quiet", action="store_true", help="Disable console log output")
    parser.add_argument("--path", default="/Users/aps/Documents/GitHub/colorDataToJson/Data", help="Folder path containing JSON file")
    parser.add_argument("--json_file", default="printer.cie.json", help="JSON file name")
    parser.add_argument(
        "--measurement-id",
        dest="measurement_id",
        help="Override MEASUREMENT_ID/measurement_id (if set, overrides JSON; else JSON value is used; else a new UUIDv4 is generated)",
    )
    parser.add_argument("--db", dest="db_uri", help="Override database connection spec + URL")
    parser.add_argument("--db-port", dest="db_port", help="Override database port")
    parser.add_argument("--db-name", dest="db_name", help="Override database name")
    parser.add_argument("--username", help="Database username (default: pgadmin)")
    parser.add_argument("--password", help="Database password (default: postgres)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and log but do not insert into database")
    parser.add_argument("--no-db-check", action="store_true", help="Skip DB connectivity check during --dry-run")
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, help="Target schema name (auto-created if missing)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Log verbosity for file/stdout handlers (default: INFO)")
    parser.add_argument("--log-file", help="Optional override path for log file (default: (source)/Logs/loadJson_YYYY-MM-DD.log)")
    parser.add_argument("--logsize", type=int, default=5, help="Max log file size in MB before rotation (default: 5MB)")
    parser.add_argument("--logbackups", type=int, default=3, help="Number of rotated backups to keep (default: 3)")
    parser.add_argument("--jsonlog", action="store_true", help="Enable JSON-formatted log output")
    parser.add_argument("--gui", action="store_true", help="Launch Tkinter GUI to enter arguments")
    args = parser.parse_args()

    # === LOG LEVEL TOGGLE ===
    log_level = logging.DEBUG if args.debug else logging.INFO
    for handler in logging.getLogger().handlers:
        handler.setLevel(log_level)
    logger.setLevel(log_level)
    # === END LOG LEVEL TOGGLE ===

    # If GUI requested (and available), collect args via GUI
    if getattr(args, "gui", False):
        if not TK_AVAILABLE:
            raise RuntimeError("Tkinter GUI requested but Tkinter is not available.")
        args = run_gui_with_prefs(args)

    log("Starting loadJson.py", logging.INFO)

    # Determine desired log level (debug flag overrides)
    desired_level = "DEBUG" if args.debug else (args.log_level if args.verbose is False else args.log_level)

    # Resolve optional --log-file (relative paths resolved against SCRIPT_DIR)
    log_override = None
    if hasattr(args, "log_file") and args.log_file:
        raw = Path(args.log_file).expanduser()
        # Treat absolute paths or (on Windows) leading backslash as absolute
        if raw.is_absolute() or str(raw).startswith("\\\\") or str(raw).startswith("\\"):
            log_override = raw
        else:
            SCRIPT_DIR = Path(__file__).resolve().parent
            log_override = (SCRIPT_DIR / raw).resolve()

    setup_logging(app_name="loadJson", log_level=desired_level, log_file_override=log_override)

    json_file = Path(args.path) / args.json_file
    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    # Determine measurement_id
    guid = None
    if args.measurement_id:
        try:
            guid = uuid.UUID(args.measurement_id)
        except ValueError:
            raise ValueError(f"Invalid UUID format: {args.measurement_id}")
    else:
        guid = None;
    

    # Override DB connection parameters from args if provided
    DB_USERNAME = args.username or DB_USERNAME
    DB_PASSWORD = args.password or DB_PASSWORD
    DB_NAME = args.db_name or DB_NAME
    DB_PORT = args.db_port or DB_PORT
    DB_URI = args.db_uri or DB_URI

    DB_URI = f"{DB_PREFIX}{DB_USERNAME}:{DB_PASSWORD}@{DB_URI}:{DB_PORT}/{DB_NAME}"
    print("DB_URI:", DB_URI)

    log(f"Using DB URI: {_redact_url(DB_URI)}", logging.INFO)
    g_schema = args.schema or DEFAULT_SCHEMA  # if empty string passed, fallback

    log(f"Using measurement_id: {guid}", logging.INFO)
    log(f"Schema: {g_schema}", logging.INFO)
    log(f"Processing file: {json_file}", logging.INFO)
    log(f"Effective measurement_id (JSON preferred): {guid}", logging.INFO)

    process_file(
        json_file=json_file,
        guid=guid,
        db_uri=DB_URI,
        dry_run=args.dry_run,
        no_db_check=args.no_db_check,
        schema=g_schema,
        force_guid=guid,
        gui=args.gui,
    )

    log("✅ All measurement records inserted successfully.", logging.INFO)


if __name__ == "__main__":
    main()