"""Abstract base classes for template processing strategies."""

from app.interfaces.template import BaseTemplateAnalyzer, BaseTemplateInjector, TemplateDocument

__all__ = [
    "BaseTemplateAnalyzer",
    "BaseTemplateInjector",
    "TemplateDocument",
]
