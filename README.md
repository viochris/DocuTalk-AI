# 📚 DocuTalk-AI (Smart Document Assistant)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Gemini%202.5%20Flash-8E75B2?logo=google&logoColor=white)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen)

## 📌 Overview
**DocuTalk-AI** is an intelligent retrieval-augmented generation (RAG) application powered by Google's **Gemini 2.5 Flash**.

Unlike basic chatbots, this system employs a **ReAct (Reasoning + Acting) Agent** architecture. It intelligently decides whether to answer questions based on your uploaded **Knowledge Base (PDF, CSV, TXT, MD)** or via **Web Search (DuckDuckGo)** if the document lacks specific information. It features incremental indexing, strict language control, and session management.

## ✨ Key Features

### 🧠 Smart RAG & Agentic Workflow
Powered by `LangChain`, the app creates a "Brain" that follows a strict priority protocol:
1.  **Priority 1:** Check Uploaded Documents (Vector Search via FAISS).
2.  **Priority 2:** Use Internal Knowledge.
3.  **Priority 3:** Web Search (Fallback if info is missing).

### ⚡ Cost-Efficient Incremental Indexing
* **Smart Processing:** The app tracks filenames (`processed_files` state). If you add a new file to an existing batch, it **only processes the new file** without re-embedding the old ones.
* **Optimization:** Saves time and API quota by appending to the existing FAISS vector store instead of rebuilding it from scratch.

### 🌐 Strict Language Enforcement
Features a custom "Translator-Researcher" prompt injection.
* **User Control:** Select **English** or **Indonesian** in the sidebar.
* **Behavior:** The AI ignores the source document's language and forces the final output into your selected language (e.g., querying an English Journal but getting the answer in Indonesian).

### 🛠️ Advanced Session Management
* **Chat History Export:** Download your entire conversation analysis as a JSON file.
* **Memory Handling:** Uses `ConversationSummaryMemory` to maintain context without overloading the token limit.
* **Safety Rails:** Robust error handling for API Quotas (429), Invalid Keys, and Empty/Corrupt Files.

### ⚙️ Customizable Intelligence
* **Creativity Slider:** Adjust the `temperature` (0.0 - 1.0) to switch between strict fact-extraction and creative summarization.
* **Visual Thinking:** Real-time visibility into the Agent's thought process (Observation -> Thought -> Action) via `StreamlitCallbackHandler`.

## 🛠️ Tech Stack
* **LLM:** Google Gemini 2.5 Flash.
* **Framework:** Streamlit.
* **Orchestration:** LangChain (Agents, Tools, Chains).
* **Vector Database:** FAISS (In-memory).
* **Embeddings:** GoogleGenerativeAIEmbeddings (`models/gemini-embedding-001`).
* **Search Tool:** DuckDuckGo Search Run.

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/viochris/DocuTalk-AI
    cd DocuTalk-AI
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run file.py
    ```

## 🚀 Usage Guide

1.  **Setup:**
    * Get your API Key from [Google AI Studio](https://aistudio.google.com/).
    * Enter the key in the sidebar.
3.  **Configuration:**
    * Choose **Response Language** (English/Indonesian).
    * Adjust **Creativity Level**.
4.  **Build Knowledge Base:**
    * Upload your documents (**PDF, CSV, MD, or TXT**).
    * Click **"🚀 Process & Embed Documents"**.
    * *Wait for the "Successfully added" toast notification.*
5.  **Chat & Analyze:**
    * Ask questions like *"Summarize the methodology"* or *"Analyze the data trends in the CSV"*.
    * Use the **Recommended Questions** expander for quick start ideas.
6.  **Export:**
    * Click **"📥 Download Chat History"** to save your insights.

## ⚠️ Limitations
* **Session Volatility:** Since FAISS is stored in RAM (Session State), refreshing the browser will clear the document index.
* **API Quotas:** Heavy usage of the Embedding model or LLM may hit Google's free tier rate limits (captured by the app's error handling).

## 📷 Gallery

### 1. Landing Interface
![Home UI](assets/home_ui.png)  
*The clean, user-friendly landing page welcoming users to DocuTalk-AI. It features a clear call-to-action for uploading documents and initializing the secure environment.*

### 2. Comprehensive Sidebar Controls
![Sidebar Settings](assets/sidebar.png)  
*The remaining part of the sidebar configuration that is not visible in the Home UI screenshot.*
### 3. Interactive Analysis & Chat
![Active Session](assets/used_ui.png)  
*The core experience in action. Once documents are processed, the chat interface enables natural language queries, displaying precise answers derived directly from the uploaded content.*

---
**Author:** [Silvio Christian, Joe](https://www.linkedin.com/in/silvio-christian-joe)
*"Unlock insights from your documents instantly."*
