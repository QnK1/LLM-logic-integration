from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


def create_llm(
    provider: str,
    model_name: str,
    api_key: str | None = None,
    temperature: float = 0.0,
    context_window: int = 8192,
) -> BaseChatModel:
    provider = provider.lower()

    if provider == "gemini":
        if not api_key:
            raise ValueError("API key required for Gemini.")
        return ChatGoogleGenerativeAI(
            model=model_name, google_api_key=api_key, temperature=temperature
        )

    elif provider == "openai":
        if not api_key:
            raise ValueError("API key required for OpenAI.")
        return ChatOpenAI(
            model=model_name, api_key=SecretStr(api_key), temperature=temperature
        )

    elif provider == "ollama":
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            format="json",
            num_ctx=context_window,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")
