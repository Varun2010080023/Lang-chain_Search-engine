# 🔎 LangChain Smart Search Assistant

An intelligent, interactive Streamlit-based search assistant built with [LangChain](https://www.langchain.com/), [Groq LLMs](https://groq.com/), and multiple real-time information tools like **DuckDuckGo**, **Wikipedia**, and **Arxiv**. Ask any question and watch the agent think step-by-step using the selected tools to provide a concise, cited response.

![LangChain Smart Search UI](assets/screenshot.png) <!-- Replace with your actual screenshot path -->

---

## 🚀 Features

- 🔍 Search across **Web (DuckDuckGo)**, **Wikipedia**, and **Arxiv**
- 🧠 Uses **LangChain Agents** for multi-tool reasoning
- 🤖 Powered by **Groq LLMs** (`llama3`, `mixtral`, etc.)
- 📡 Live agent "thoughts" displayed using `StreamlitCallbackHandler`
- ⚙️ Customizable model, temperature, max tokens, and iterations
- 💬 Chat-style interface with persistent message history

---

## 🛠️ Installation

1. **Clone this repository**

```bash
git clone https://github.com/your-username/langchain-smart-search.git
cd langchain-smart-search
