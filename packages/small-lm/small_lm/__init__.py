"""Small models on phones and laptops: quantise, retrieve locally, and route by calibrated confidence."""

from small_lm.local import Local
from small_lm.router import Answer, Router

__all__ = ["Answer", "Local", "Router"]
__version__ = "0.6.0"
