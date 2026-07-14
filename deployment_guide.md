# Deployment Guide: Universal RAG System

Follow this guide to deploy your Universal RAG System to Streamlit Community Cloud.

## Streamlit Community Cloud (Free & Direct from GitHub)

Streamlit Community Cloud hosts apps directly from your GitHub repository.

### Step 1: Upload Your Code to GitHub
1. Create a repository on your GitHub account (e.g., `universal-rag-system`).
2. Run these commands in your local project terminal:
   ```bash
   git init
   git add .
   git commit -m "Initialize Universal RAG System"
   git branch -M main
   git remote add origin https://github.com/your-username/universal-rag-system.git
   git push -u origin main
   ```
   *The `.gitignore` we created will automatically prevent virtual environments (`venv/`), database files (`chroma_db/`), and env secrets (`.env`) from being uploaded to GitHub.*

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in using your GitHub account.
2. Click **New app** (or **Create app**).
3. Fill out the deployment form:
   - **Repository**: Choose `your-username/universal-rag-system`.
   - **Branch**: Select `main`.
   - **Main file path**: Set to `app.py`.
4. Click **Deploy!**

### Step 3: Configure API Keys (Secrets)
1. Once your app is running, click the **Settings** gear icon in the bottom-right corner.
2. Select **Secrets** in the settings panel.
3. Paste the following configuration, substituting your actual key:
   ```toml
   GROQ_API_KEY = "gsk_your_actual_groq_api_key_goes_here"
   ```
4. Click **Save**. The app will automatically rebuild and run.
