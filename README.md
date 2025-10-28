
---

# 🤖 Customer Support Agent – Streamlit + Gemini + ChromaDB

**An AI-powered customer support assistant** built with **Google Gemini**, **LangChain**, and **ChromaDB** — enabling users to upload custom documents and chat with them in natural language.

### 🚀 Live Demo
[Try it on Streamlit →](https://customer-support-agent-ai.streamlit.app/)

---

## 🧠 Overview

This project implements a **retrieval-augmented generation (RAG)** chatbot that can act as a company’s support agent.
Users can either use a **default demo knowledge base** or upload their own text files to train the chatbot on-the-fly.

* 🧩 **Multi-user architecture** — isolated document stores per user session
* ⚡ **Fast streaming responses** — Gemini API with optimized prompt chaining
* 🧹 **Automatic cleanup** — removes temporary vector stores after user sessions
* 💾 **Persistent Chroma vector DB** — for efficient semantic search
* 💬 **Context-aware Q&A** — answers derived directly from uploaded documents

---

## 🏗️ Tech Stack

| Layer                  | Technology                                                |
| ---------------------- | --------------------------------------------------------- |
| **Frontend / UI**      | [Streamlit](https://streamlit.io)                         |
| **LLM Model**          | [Google Gemini (gemini-2.5-flash)](https://ai.google.dev) |
| **Embeddings**         | `GoogleGenerativeAIEmbeddings`                            |
| **Vector Store**       | [ChromaDB](https://www.trychroma.com)                     |
| **Orchestration**      | LangChain + (simplified) LangGraph                        |
| **Environment Config** | `python-dotenv`                                           |

---

## 🧩 Architecture

```
User → Streamlit UI → Gemini Chat Model
                     ↘︎
                      Vector Store (ChromaDB)
                        ↑
              Uploaded or Default Documents
```

Each user session is sandboxed:

* Unique UUID-based Chroma collection (`user_docs_<uuid>`)
* Cleaned automatically when session ends
* Default demo collection shared across all users

---

## 🛠️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/support-agent.git
cd support-agent
```

### 2️⃣ setup a virtual environment

```bash
uv sync
```

### 3️⃣ Add your environment variable

Create a `.env` file in the root directory:

```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4️⃣ Run the app

```bash
streamlit run app.py
```

---

## 💡 Key Features

* 🔍 **Document Upload & Indexing**

  * Upload multiple `.txt` files per session
  * Indexed instantly with Gemini embeddings

* 🤝 **Contextual Chat**

  * Uses chat history + retrieved context for relevant answers

* ⚡ **Performance Optimizations**

  * Streamed Gemini responses
  * Simplified graph flow for minimal latency

* 🧹 **Auto Cleanup**

  * Per-session collections removed safely when session closes
  * Startup routine cleans up stale collections

---

## 🧰 Example Use Cases

* Customer support knowledge base
* Internal policy document assistant
* Product manual Q&A bot
* Company FAQ automation

---

## 🌐 Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub.

2. Go to [Streamlit Cloud](https://streamlit.io/cloud) → “New app”.

3. Select your repo and set environment variable:

   ```
   GOOGLE_API_KEY=your_gemini_api_key
   ```

4. Deploy — each visitor gets an isolated chat session!

---

## ⭐ Future Enhancements

* Support for PDFs and web URLs
* Add RAG feedback loop (source citations)
* Authentication for enterprise use
* Cloud object storage for persistent user data

---

## 🪪 License

This project is licensed under the MIT License — feel free to fork and extend.

---

