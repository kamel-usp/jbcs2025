import os
from typing import Optional
from dotenv import load_dotenv
from omegaconf import OmegaConf

# Load environment variables from .env file
load_dotenv()


def register_resolvers():
    """
    Register custom OmegaConf resolvers for environment variables and other interpolations.
    Must be called before any configuration is loaded.
    """
    # Register environment variable resolver if not already registered
    if not OmegaConf.has_resolver("env"):
        OmegaConf.register_new_resolver(
            "env", 
            lambda name, default=None: os.environ.get(name, default)
        )


def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for the specified provider from environment variables.
    
    Args:
        provider: The API provider ('maritaca_sabia_3', 'openai_chatgpt_4o', etc.)
        
    Returns:
        The API key if found, None otherwise
    """
    provider_map = {
        "maritaca_sabia_3": "MARITACA_API_KEY",
        "openai_chatgpt_4o": "OPENAI_API_KEY",
        "deepseek_r1": "DEEPSEEK_API_KEY"
        # Add more providers as needed
    }
    
    env_var = provider_map.get(provider)
    if not env_var:
        return None
        
    api_key = os.environ.get(env_var)
    return api_key