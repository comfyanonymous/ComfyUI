

class GetChoices:

    def __init__(self, resources, configs):
        self.resources = resources
        self.configs = configs

    def get_choices(self, source: str):
        """
        Get choices of things from certain websites
        
        """




# Create a list of available models for the API
AVAILABLE_MODELS: list[str] = []
ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    "claude-3-opus-latest",
]
OPEN_AI_MODELS = [
    "gpt-4o",
    "chatgpt-4o-latest",
    "gpt-4o-mini",
    "o1",
    "o1-mini",
    "o1-preview",
    "gpt-4o-realtime-preview",
    "gpt-4o-mini-realtime-preview",
    "gpt-4o-audio-preview",
]
AVAILABLE_MODELS.extend(ANTHROPIC_MODELS)
AVAILABLE_MODELS.extend(OPEN_AI_MODELS)


OPEN_AI_EMBEDDING_MODELS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
]
TEXT_EMBEDDING_MODELS = []
TEXT_EMBEDDING_MODELS.append(OPEN_AI_EMBEDDING_MODELS)