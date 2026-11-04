"""Environment-backed settings. Secrets are held as ``SecretStr`` and never logged."""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============= Settings =============


class Settings(BaseSettings):
    """Environment variables read by the platform; each field is optional until a stage needs it."""

    model_config = SettingsConfigDict(extra="ignore")

    hf_token: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    hf_namespace: str | None = None
    platform_root: str | None = None
    mlflow_tracking_uri: str | None = None
    llama_cpp_dir: str | None = None

    def require_groq_key(self) -> str:
        """Return the Groq API key or raise a descriptive error."""
        if self.groq_api_key is None:
            raise RuntimeError("GROQ_API_KEY is not set in the environment")
        return self.groq_api_key.get_secret_value()

    def require_hf_token(self) -> str:
        """Return the Hugging Face token or raise a descriptive error."""
        if self.hf_token is None:
            raise RuntimeError("HF_TOKEN is not set in the environment")
        return self.hf_token.get_secret_value()

    def require_hf_namespace(self) -> str:
        """Return the Hub namespace that owns published repositories."""
        if not self.hf_namespace:
            raise RuntimeError("HF_NAMESPACE is not set in the environment")
        return self.hf_namespace


def get_settings(env_file: str | None = None) -> Settings:
    """
    Build settings from the process environment and, optionally, an explicit env file.

    Parameters
    ----------
    env_file : str, optional
        Path of a dotenv file. No file is read unless this is given.
    """
    return Settings(_env_file=env_file)  # type: ignore[call-arg]
