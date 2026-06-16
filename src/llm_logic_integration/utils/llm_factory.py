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
    context_window: int = 32768,
    max_tokens: int | None = None,
) -> BaseChatModel:
    provider = provider.lower()

    if provider == "gemini":
        if not api_key:
            raise ValueError("API key required for Gemini.")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    elif provider == "openai":
        if not api_key:
            raise ValueError("API key required for OpenAI.")
        return ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            temperature=temperature,
        )

    elif provider == "ollama":
        kwargs = {
            "model": model_name,
            "temperature": temperature,
            "num_ctx": context_window,
            "client_kwargs": {
                "timeout": 120.0,
            },
        }

        if max_tokens is not None:
            kwargs["num_predict"] = max_tokens

        return ChatOllama(**kwargs)  # ty:ignore[invalid-argument-type]

    else:
        raise ValueError(f"Unsupported provider: {provider}")
