# MEMORA — AI-Based Personal File Search System

**Remember. Search. Find.**

## Overview

MEMORA is an AI-based personal file search system that helps users find files using their content instead of relying only on filenames. It extracts text from supported files and uses semantic search to retrieve relevant results from natural-language queries.

## Problem Statement

Finding a particular screenshot or document becomes difficult when users forget its filename or storage location. Traditional file searches often depend on filenames and exact keywords. MEMORA aims to make file discovery easier by searching the extracted content of supported files.

## Features

- Folder selection and scanning
- OCR-based text extraction from images
- Text extraction from PDF and TXT files
- Semantic search using Sentence Transformers
- Similarity-based ranking of search results
- File filtering and result previews
- Local SQLite storage for extracted text and file details
- Streamlit-based user interface

## Technology Stack

- **Language:** Python
- **Interface:** Streamlit
- **OCR:** Tesseract OCR, Pytesseract
- **Semantic embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Similarity search:** FAISS
- **Database:** SQLite
- **PDF processing:** PyMuPDF
- **Image processing:** Pillow

## Supported File Formats

- PDF (`.pdf`)
- PNG (`.png`)
- JPG/JPEG (`.jpg`, `.jpeg`)
- Text files (`.txt`)

## Setup and Installation

### Prerequisites

- Python 3.11
- Tesseract OCR installed on the system
- Git (optional, for cloning the repository)

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
```

Replace the placeholders with your actual repository URL and folder name.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Tesseract OCR

Install Tesseract OCR separately and configure its executable path in the application if required.

### 5. Run MEMORA

```bash
streamlit run app.py
```

The application will open in your browser.

## How to Use

1. Open MEMORA.
2. Select the folder containing your files.
3. Scan the selected folder.
4. Enter a natural-language query describing the information you remember.
5. Review the ranked results and open or preview relevant files.

## Current Limitations

- Only the listed file formats are currently supported.
- Search quality depends on the text extracted from files.
- Image OCR may not accurately extract text from every image.
- Users must initiate folder scanning manually.
- Search results depend on the available indexed content.

## Future Scope

- Automatic monitoring of newly added or modified files
- Desktop-based quick search interface
- Support for additional file formats
- Smart file categorization
- Voice-based search
- Mobile application
- Improved search accuracy and performance

## Privacy

MEMORA is designed for local file search. File content is processed by the local application; the project should be configured and reviewed to ensure that no private files or extracted content are uploaded to external services.

## Project Status

Academic project developed as part of the B.Tech Computer Science and Engineering curriculum.