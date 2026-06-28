import os
import yaml
from pathlib import Path

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

class Config:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def project_name(self) -> str:
        return self._config["project"]["name"]

    @property
    def tickers(self) -> list:
        return self._config["stock_universe"]["tickers"]

    @property
    def database_path(self) -> Path:
        # Resolve database path relative to project root
        db_path = self._config["data"]["database_path"]
        return (self.project_root / db_path).resolve()

    @property
    def raw_dir(self) -> Path:
        return (self.project_root / self._config["data"]["raw_dir"]).resolve()

    @property
    def prices_dir(self) -> Path:
        return (self.project_root / self._config["data"]["prices_dir"]).resolve()

    @property
    def news_dir(self) -> Path:
        return (self.project_root / self._config["data"]["news_dir"]).resolve()

    @property
    def transcripts_dir(self) -> Path:
        return (self.project_root / self._config["data"]["transcripts_dir"]).resolve()

    @property
    def start_years_ago(self) -> int:
        return self._config["ingestion"]["start_years_ago"]

    @property
    def retry_attempts(self) -> int:
        return self._config["ingestion"]["retry_attempts"]

    @property
    def retry_backoff_factor(self) -> float:
        return float(self._config["ingestion"]["retry_backoff_factor"])

    @property
    def retry_max_backoff(self) -> float:
        return float(self._config["ingestion"]["retry_max_backoff"])

    @property
    def log_file(self) -> Path:
        return (self.project_root / self._config["logging"]["log_file"]).resolve()

    @property
    def log_level(self) -> str:
        return self._config["logging"]["log_level"]

    @property
    def finnhub_api_key(self) -> str:
        # Check env first, then fall back to config file
        return os.environ.get("FINNHUB_API_KEY") or self._config["api_keys"]["finnhub_api_key"]

    @property
    def news_api_key(self) -> str:
        # Check env first, then fall back to config file
        return os.environ.get("NEWS_API_KEY") or self._config["api_keys"]["news_api_key"]

    @property
    def fred_api_key(self) -> str:
        # Check env first, then fall back to config file
        return os.environ.get("FRED_API_KEY") or self._config["api_keys"]["fred_api_key"]

# Singleton instance
config = Config()
