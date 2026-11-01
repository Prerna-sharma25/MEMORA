
import os
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pytesseract
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MEMORA | AI File Search",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "digital_memory.db"

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# =========================================================
# PREMIUM DARK UI
# =========================================================

st.markdown("""
<style>
:root {
    --bg: #0b0915;
    --panel: #141125;
    --panel2: #1b1631;
    --border: #30264b;
    --purple: #a18aff;
    --pink: #e6a0db;
    --text: #f5f1ff;
    --muted: #aaa2c5;
}

.stApp {
    background:
        radial-gradient(ellipse at 75% 0%, #211638 0%, transparent 38%),
        #0b0915;
    color: var(--text);
}

.block-container {
    max-width: 1350px;
    padding: 2rem 2.5rem 3rem;
}

[data-testid="stSidebar"] {
    background: #100d1e;
    border-right: 1px solid #29213e;
}

[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.7rem;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 4px 0 5px;
}

.brand-icon {
    font-size: 35px;
}


.brand-name {
    color: #f7f3ff;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: 0;
}


.brand-sub {
    color: #968bac;
    font-size: 12px;
    margin-left: 49px;
    margin-bottom: 30px;
}

.sidebar-note {
    background: #19142a;
    border: 1px solid #302448;
    border-radius: 14px;
    padding: 15px;
    color: #bcb3d5;
    font-size: 13px;
    line-height: 1.7;
}

.hero {
    padding: 6px 0 22px;
}

.eyebrow {
    color: #b6a4ff;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
    margin-left: 60px;
}

.hero h2 {
    font-family: Georgia, "Times New Roman", serif;
    font-size: clamp(32px, 3.2vw, 44px);
    line-height: 1.12;
    color: #f8f5ff;
    margin: 0 0 13px;
}

.hero p {
    color: #aaa1c4;
    font-size: 16px;
    line-height: 1.7;
    margin: 0;
}

.section-heading {
    color: #f1ecff;
    font-size: 23px;
    font-weight: 650;
    margin: 14px 0 5px;
}

.section-sub {
    color: #a79dbd;
    font-size: 14px;
    margin-bottom: 16px;
}

div[data-testid="stTextInput"] input {
    background: #151126 !important;
    color: #ffffff !important;
    border: 1px solid #51416e !important;
    border-radius: 14px !important;
    min-height: 51px;
}

div[data-testid="stTextInput"] input:focus {
    border-color: #a18aff !important;
    box-shadow: 0 0 0 1px #a18aff !important;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(110deg, #a18aff, #dca0df) !important;
    color: #171126 !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 43px;
    font-weight: 700 !important;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    filter: brightness(1.1);
    border: none !important;
    box-shadow: 0 0 20px #9e83ff30;
}

div[data-testid="stCheckbox"] label {
    color: #e7e0f7 !important;
}

div[data-testid="stCheckbox"] {
    padding: 4px 0;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: linear-gradient(145deg, #171329, #121020);
    border: 1px solid #302644 !important;
    border-radius: 17px !important;
}

.file-count {
    background: #1b1630;
    border: 1px solid #3a2e58;
    border-radius: 13px;
    padding: 12px 16px;
    color: #c7b8ff;
    font-size: 14px;
    text-align: center;
}

.result-title {
    color: #f4efff;
    font-size: 18px;
    font-weight: 700;
    overflow-wrap: anywhere;
    margin-bottom: 6px;
}

.result-path {
    color: #a69abf;
    font-size: 12px;
    overflow-wrap: anywhere;
    line-height: 1.6;
}

.score {
    display: inline-block;
    color: #d5c8ff;
    background: #2a2144;
    border: 1px solid #463660;
    padding: 7px 10px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 700;
    white-space: nowrap;
}

.result-type {
    color: #b9a9f4;
    font-size: 13px;
    margin-bottom: 10px;
}

.footer {
    text-align: center;
    color: #716783;
    font-size: 12px;
    margin-top: 45px;
    padding-top: 20px;
    border-top: 1px solid #272039;
}

hr {
    border-color: #302644 !important;
}

@media (max-width: 700px) {
    .block-container {
        padding: 1.2rem 1rem 2rem;
    }
    .hero h1 {
        font-size: 36px;
    }
}






/* Main page MEMORA heading - separate from sidebar */
.main-brand {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 22px;
}

.main-brand-icon {
    font-size: 44px;
    line-height: 1;
}

.main-brand-name {
    color: #f8f5ff;
    font-family: Georgia, "Times New Roman", serif;
    font-size: 52px;
    font-weight: 700;
    line-height: 1.1;
    letter-spacing: 0;
}

/* Hide only the Deploy button */
.stAppDeployButton {
    display: none !important;
}

/* Keep the header and sidebar toggle visible */
.stAppHeader,
header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

[data-testid="stSidebarCollapsedControl"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}


</style>
""", unsafe_allow_html=True)


# =========================================================
# DATABASE
# =========================================================

def create_database():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                filepath TEXT UNIQUE,
                extracted_text TEXT
            )
        """)


create_database()


# =========================================================
# AI MODEL AND SEARCH INDEX
# =========================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()


@st.cache_data(show_spinner=False)
def create_search_index(texts):
    embeddings = model.encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=False
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.asarray(embeddings, dtype="float32"))

    return index


# =========================================================
# IMAGE PREVIEW
# =========================================================

@st.dialog("Image Preview", width="large")
def show_image_preview(image_path):
    try:
        with Image.open(image_path) as image:
            st.image(image, use_container_width=True)

        st.caption(image_path)

        if st.button("Close preview"):
            st.session_state["memora_selected_image"] = None
            st.rerun()

    except Exception:
        st.error("Unable to open this image.")


# =========================================================
# OPEN ORIGINAL FILE
# =========================================================

def open_original_file(file_path):
    try:
        path = Path(file_path)

        if not path.is_file():
            st.error("Original file not found.")
            return

        os.startfile(str(path.resolve()))

    except Exception as error:
        st.error(f"Could not open file: {error}")


# =========================================================
# COMMON WINDOWS FOLDERS
# =========================================================

def find_existing_folder(*parts):
    candidates = []
    onedrive = os.environ.get("OneDrive")

    if onedrive:
        candidates.append(Path(onedrive).joinpath(*parts))

    candidates.append(Path.home().joinpath("OneDrive", *parts))
    candidates.append(Path.home().joinpath(*parts))

    checked = set()

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            key = str(resolved).lower()

            if key in checked:
                continue

            checked.add(key)

            if resolved.is_dir():
                return resolved

        except (OSError, PermissionError):
            continue

    return None


folder_options = {
    "Screenshots": ("Pictures", "Screenshots"),
    "Downloads": ("Downloads",),
    "Documents": ("Documents",),
    "Desktop": ("Desktop",),
    "Pictures": ("Pictures",),
}


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("""
    <div class="brand">
        <span class="brand-icon">🧠</span>
        <span class="brand-name">MEMORA</span>
    </div>
    <div class="brand-sub">YOUR PERSONAL FILE MEMORY</div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["🏠  Home", "⚙️  Settings", "ℹ️  About"],
        label_visibility="collapsed",
        key="memora_navigation"
    )

    st.markdown("---")

    st.markdown("""
    <div class="sidebar-note">
        <b>🔒 Local-first search</b><br>
        Your files are indexed in your local
        database. MEMORA searches their
        extracted content to help you find
        what you remember.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with sqlite3.connect(DB_PATH) as conn:
        sidebar_count = conn.execute(
            "SELECT COUNT(*) FROM files"
        ).fetchone()[0]

    st.caption(f"🧠 {sidebar_count:,} files remembered")
    st.caption("MEMORA · AI File Search")



# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠  Home":

    st.markdown("""
    <div class="main-brand">
        <span class="main-brand-icon">🧠</span>
        <span class="main-brand-name">MEMORA</span>
    </div>

    <div class="hero">
        <div class="eyebrow">Remember. Search. Find.</div>
        <h2>Find what you remember.</h2>
        <p>
            Search your files by what they say, not what you named them.
            <br>
            Describe a memory. Let AI help you find the file.
        </p>
    </div>
    """, unsafe_allow_html=True)



    # -----------------------------------------------------
    # FOLDER SELECTION
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-heading">📁 Choose your folders</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-sub">'
        'Select the locations where MEMORA should look for your files.'
        '</div>',
        unsafe_allow_html=True
    )

    folder_paths = {
        name: find_existing_folder(*parts)
        for name, parts in folder_options.items()
    }

    folder_icons = {
        "Screenshots": "🖼️",
        "Downloads": "📥",
        "Documents": "📄",
        "Desktop": "🖥️",
        "Pictures": "🌄",
    }

    folder_names = list(folder_paths.items())

    for start in range(0, len(folder_names), 3):
        group = folder_names[start:start + 3]
        cols = st.columns(3)

        for col, (name, folder) in zip(cols, group):
            with col:
                with st.container(border=True):
                    st.markdown(
                        f"### {folder_icons[name]} {name}"
                    )

                    if folder is None:
                        st.caption("Folder not found")
                        st.checkbox(
                            "Unavailable",
                            value=False,
                            disabled=True,
                            key=f"memora_folder_{name}"
                        )
                    else:
                        st.caption("Available on this device")
                        st.checkbox(
                            "Include folder",
                            key=f"memora_folder_{name}"
                        )

    selected_folders = [
        folder
        for name, folder in folder_paths.items()
        if folder is not None
        and st.session_state.get(f"memora_folder_{name}", False)
    ]

    # -----------------------------------------------------
    # SCAN CONTROLS
    # -----------------------------------------------------

    st.markdown("<br>", unsafe_allow_html=True)

    scan_col, count_col = st.columns([1, 1])

    with scan_col:
        scan_button = st.button(
            "✦  Scan selected folders",
            use_container_width=True
        )

    with count_col:
        with sqlite3.connect(DB_PATH) as conn:
            total_files = conn.execute(
                "SELECT COUNT(*) FROM files"
            ).fetchone()[0]

        st.markdown(
            f'<div class="file-count">'
            f'🧠 &nbsp; {total_files:,} files remembered'
            f'</div>',
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # SCANNING
    # -----------------------------------------------------

    if scan_button:
        if not selected_folders:
            st.warning("Please select at least one available folder.")

        else:
            supported_extensions = {
                ".pdf", ".png", ".jpg", ".jpeg", ".txt"
            }

            files_by_path = {}

            for folder in selected_folders:
                try:
                    for file in folder.rglob("*"):
                        try:
                            if (
                                file.is_file()
                                and file.suffix.lower() in supported_extensions
                            ):
                                files_by_path[str(file.resolve())] = file
                        except (OSError, PermissionError):
                            continue
                except (OSError, PermissionError):
                    continue

            files = list(files_by_path.values())

            if not files:
                st.warning("No supported files found in selected folders.")

            else:
                progress = st.progress(0)
                status = st.empty()
                processed = 0
                skipped = 0

                try:
                    with sqlite3.connect(DB_PATH) as conn:
                        for i, file in enumerate(files):
                            try:
                                extension = file.suffix.lower()
                                extracted_text = ""

                                if extension in {".png", ".jpg", ".jpeg"}:
                                    with Image.open(file) as image:
                                        extracted_text = (
                                            pytesseract.image_to_string(image)
                                        )

                                elif extension == ".txt":
                                    extracted_text = file.read_text(
                                        encoding="utf-8",
                                        errors="ignore"
                                    )

                                elif extension == ".pdf":
                                    import fitz

                                    with fitz.open(file) as pdf:
                                        extracted_text = "".join(
                                            p.get_text() for p in pdf
                                        )

                                conn.execute("""
                                    INSERT OR REPLACE INTO files
                                    (filename, filepath, extracted_text)
                                    VALUES (?, ?, ?)
                                """, (
                                    file.name,
                                    str(file.resolve()),
                                    extracted_text
                                ))

                                processed += 1

                            except Exception:
                                skipped += 1

                            if (i + 1) % 10 == 0 or i + 1 == len(files):
                                conn.commit()

                            progress.progress((i + 1) / len(files))
                            status.text(
                                f"Processing {i + 1} of {len(files)} files..."
                            )

                except Exception as error:
                    st.error(f"Scan error: {error}")

                status.empty()
                create_search_index.clear()

                st.success(
                    f"Scan complete! {processed} files processed; "
                    f"{skipped} files skipped."
                )
                st.rerun()

    # -----------------------------------------------------
    # SEARCH
    # -----------------------------------------------------

    st.markdown("---")

    st.markdown(
        '<div class="section-heading">✨ Ask your memory</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="section-sub">'
        'Describe a topic, phrase, or detail you remember from the file.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form("memora_search_form"):
        query_col, button_col = st.columns([5, 1.3])

        with query_col:
            query = st.text_input(
                "Search",
                placeholder="e.g. the screenshot about salary prediction",
                label_visibility="collapsed"
            )

        with button_col:
            search_button = st.form_submit_button(
                "🔎 Search",
                use_container_width=True
            )

    if search_button:
        st.session_state["memora_selected_image"] = None
        st.session_state["memora_active_query"] = query.strip()

    active_query = st.session_state.get("memora_active_query", "")

    # -----------------------------------------------------
    # SEARCH RESULTS WITH FOLDER FILTERING
    # -----------------------------------------------------

    if active_query:
        with sqlite3.connect(DB_PATH) as conn:
            all_rows = conn.execute("""
                SELECT filename, filepath, extracted_text
                FROM files
                WHERE extracted_text IS NOT NULL
                  AND TRIM(extracted_text) != ''
            """).fetchall()

        selected_roots = []

        for folder in selected_folders:
            try:
                selected_roots.append(folder.resolve())
            except (OSError, PermissionError):
                continue

        rows = []

        for row in all_rows:
            try:
                saved_path = Path(row[1]).resolve()

                if any(
                    saved_path.is_relative_to(root)
                    for root in selected_roots
                ):
                    rows.append(row)

            except (OSError, PermissionError, ValueError):
                continue

        if not selected_roots:
            st.info("Select at least one folder above to search within it.")

        elif not rows:
            st.warning(
                "No saved files found in the selected folders. "
                "If these folders have not been scanned before, "
                "click 'Scan selected folders' first."
            )

        else:
            filenames = [row[0] for row in rows]
            filepaths = [row[1] for row in rows]
            texts = [row[2] for row in rows]

            with st.spinner("Searching your files..."):
                index = create_search_index(tuple(texts))

                query_embedding = model.encode(
                    [active_query],
                    normalize_embeddings=True
                )

                number_of_results = min(3, len(texts))

                scores, indices = index.search(
                    np.asarray(query_embedding, dtype="float32"),
                    number_of_results
                )

            st.markdown(
                f'<div class="section-heading">'
                f'🔍 Search results</div>',
                unsafe_allow_html=True
            )
            st.caption(
                f"Showing up to {number_of_results} relevant files"
            )

            for rank, (score, idx) in enumerate(
                zip(scores[0], indices[0]),
                start=1
            ):
                if idx < 0:
                    continue

                file_path = filepaths[idx]
                filename = filenames[idx]
                extension = Path(file_path).suffix.lower()

                with st.container(border=True):
                    left, middle, right = st.columns([1, 3.5, 1.2])

                    with left:
                        if extension in {".png", ".jpg", ".jpeg"}:
                            try:
                                with Image.open(file_path) as image:
                                    st.image(image, use_container_width=True)
                            except Exception:
                                st.markdown("## 🖼️")
                        elif extension == ".pdf":
                            st.markdown("## 📄")
                            st.caption("PDF document")
                        elif extension == ".txt":
                            st.markdown("## 📝")
                            st.caption("Text document")
                        else:
                            st.markdown("## 📁")

                    with middle:
                        st.markdown(
                            f'<div class="result-title">'
                            f'{filename}</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f'<div class="result-path">'
                            f'{file_path}</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f'<div class="result-type">'
                            f'{extension[1:].upper()} file · '
                            f'Result {rank}</div>',
                            unsafe_allow_html=True
                        )

                    with right:
                        st.markdown(
                            f'<div class="score">'
                            f'✦ {score:.3f}</div>',
                            unsafe_allow_html=True
                        )

                        if extension in {".png", ".jpg", ".jpeg"}:
                            if st.button(
                                "View image",
                                key=f"view_image_{idx}_{rank}",
                                use_container_width=True
                            ):
                                st.session_state[
                                    "memora_selected_image"
                                ] = file_path
                                st.rerun()

                            if st.button(
                                "Open original",
                                key=f"open_image_{idx}_{rank}",
                                use_container_width=True
                            ):
                                open_original_file(file_path)

                        elif extension == ".pdf":
                            if st.button(
                                "Open PDF",
                                key=f"open_pdf_{idx}_{rank}",
                                use_container_width=True
                            ):
                                open_original_file(file_path)

                        elif extension == ".txt":
                            if st.button(
                                "Open file",
                                key=f"open_txt_{idx}_{rank}",
                                use_container_width=True
                            ):
                                open_original_file(file_path)

    selected_image = st.session_state.get("memora_selected_image")

    if selected_image:
        show_image_preview(selected_image)


# =========================================================
# SETTINGS PAGE
# =========================================================

elif page == "⚙️  Settings":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">PREFERENCES</div>
        <h1>Settings</h1>
        <p>Manage how you use MEMORA on this device.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "MEMORA currently uses your selected local folders, "
        "SQLite database, and local semantic-search pipeline."
    )

    st.markdown("### Supported file formats")
    st.write("PDF, PNG, JPG, JPEG and TXT")

    st.markdown("### Search model")
    st.write("SentenceTransformer · all-MiniLM-L6-v2")

    st.markdown("### Database")
    st.code(str(DB_PATH))


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "ℹ️  About":
    st.markdown("""
    <div class="hero">
        <div class="eyebrow">ABOUT THE PROJECT</div>
        <h1>Meet MEMORA.</h1>
        <p>
            Your personal AI-powered file search system.
            Find files by describing what you remember.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Why MEMORA?")
    st.write(
        "Sometimes we remember what a screenshot or document contains, "
        "but not its filename. MEMORA extracts text from supported files "
        "and uses semantic similarity to help locate relevant results."
    )

    st.markdown("### Built with")
    st.write(
        "Python · Streamlit · SentenceTransformers · FAISS · SQLite · "
        "Tesseract OCR · PyMuPDF"
    )

    st.markdown("### Privacy")
    st.write(
        "Files are processed by this local application and indexed in "
        "its local SQLite database. Search uses the extracted text."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer">
    🧠 MEMORA &nbsp;·&nbsp; Remember. Search. Find.<br>
    Local-first AI semantic file search
</div>
""", unsafe_allow_html=True)
