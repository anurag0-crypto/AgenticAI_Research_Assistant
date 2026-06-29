"""
llm.py — one switch to change which LLM powers every agent role.

Set LLM_PROVIDER in .env to "anthropic", "openai", "groq", "gemini", or
"cerebras" and every node in graph.py picks it up automatically through
get_llm(). This is what satisfies the "use an LLM API (Gemini, OpenAI,
Claude, Groq, etc.)" requirement without hard-coding a single vendor
anywhere else in the app.

Each branch only imports its provider package when it's actually selected,
so you don't need all SDKs installed — just the one you use.
"""

import os

# langchain-google-genai reads GOOGLE_API_KEY by default; we let students set
# the more obvious GEMINI_API_KEY in .env and bridge it here.
if os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


def get_llm(temperature: float = 0.3):
    """Return a LangChain chat model for whichever provider is configured."""
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            temperature=temperature,
            max_tokens=4096,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            temperature=temperature,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            temperature=temperature,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-3-flash"),
            temperature=temperature,
        )

    if provider == "cerebras":
        from langchain_cerebras import ChatCerebras

        # ChatCerebras reads CEREBRAS_API_KEY from the environment
        # automatically — no need to pass it explicitly.
        return ChatCerebras(
            model=os.getenv("CEREBRAS_MODEL", "llama-3.3-70b"),
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. "
        "Use one of: anthropic, openai, groq, gemini, cerebras."
    )