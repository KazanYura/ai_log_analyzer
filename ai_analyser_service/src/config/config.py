from pydantic import BaseSettings

DEFAULT_MODEL_NAME = "microsoft/DialoGPT-medium"

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    model_name: str = "google/flan-t5-small"
    max_input_length: int = 512
    max_output_length: int = 256
    num_beams: int = 4
    no_repeat_ngram_size: int = 2
    
    class Config:
        env_file = ".env"
        env_prefix = "AI_ANALYZER_"


settings = Settings()
