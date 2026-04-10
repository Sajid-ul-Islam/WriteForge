from abc import ABC, abstractmethod


class AIProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str, temperature: float = 0.7) -> str:
        pass

    @abstractmethod
    def name(self) -> str:
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


def get_provider(provider_name: str, **kwargs) -> AIProvider:
    providers = {
        "OpenAI": OpenAIProvider,
        "Gemini": GeminiProvider,
        "Ollama": OllamaProvider,
    }
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
    return providers[provider_name](**kwargs)
