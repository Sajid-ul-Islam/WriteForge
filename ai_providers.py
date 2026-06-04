from abc import ABC, abstractmethod


class AIProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str, temperature: float = 0.7) -> str:
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def validate(self) -> bool:
        pass


class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

    def name(self) -> str:
        return f"OpenAI ({self.model})"

    def validate(self) -> bool:
        self.client.models.list()
        return True


class GeminiProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model,
            system_instruction=None,
        )
        self.model_name = model

    def generate(self, prompt: str, system_prompt: str, temperature: float = 0.7) -> str:
        import google.generativeai as genai
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system_prompt,
            generation_config=genai.GenerationConfig(temperature=temperature),
        )
        response = model.generate_content(prompt)
        return response.text

    def name(self) -> str:
        return f"Gemini ({self.model_name})"

    def validate(self) -> bool:
        import google.generativeai as genai
        for _ in genai.list_models():
            break
        return True


class AnthropicProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20240620"):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str, temperature: float = 0.7) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return message.content[0].text

    def name(self) -> str:
        return f"Anthropic ({self.model})"

    def validate(self) -> bool:
        self.client.models.list()
        return True


class GroqProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str, temperature: float = 0.7) -> str:
        chat_completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=temperature,
        )
        return chat_completion.choices[0].message.content

    def name(self) -> str:
        return f"Groq ({self.model})"

    def validate(self) -> bool:
        self.client.models.list()
        return True


class OpenRouterProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "google/gemini-2.0-flash-001"):
        from openai import OpenAI
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model = model

    def generate(self, prompt: str, system_prompt: str, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            extra_headers={
                "HTTP-Referer": "https://writeforge.ai",
                "X-Title": "WriteForge AI",
            }
        )
        return response.choices[0].message.content

    def name(self) -> str:
        return f"OpenRouter ({self.model})"

    def validate(self) -> bool:
        self.client.models.list()
        return True


class CustomOpenAIProvider(AIProvider):
    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, prompt: str, system_prompt: str, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content

    def name(self) -> str:
        return f"Custom ({self.model})"

    def validate(self) -> bool:
        self.client.models.list()
        return True


class OllamaProvider(AIProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, prompt: str, system_prompt: str, temperature: float = 0.7) -> str:
        import requests
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def name(self) -> str:
        return f"Ollama ({self.model})"

    def validate(self) -> bool:
        import requests
        response = requests.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        return True


def get_provider(provider_name: str, **kwargs) -> AIProvider:
    providers = {
        "OpenAI": OpenAIProvider,
        "Gemini": GeminiProvider,
        "Anthropic": AnthropicProvider,
        "Groq": GroqProvider,
        "OpenRouter": OpenRouterProvider,
        "Ollama": OllamaProvider,
        "Custom": CustomOpenAIProvider,
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
    return providers[provider_name](**kwargs)
