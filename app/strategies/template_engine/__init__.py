"""Template engine strategies.

Implements template analysis and Jinja2 tag injection for Word documents.
"""

from app.strategies.template_engine.analyzer import TemplateAnalyzer
from app.strategies.template_engine.injector import TemplateInjector

__all__ = [
    "TemplateAnalyzer",
    "TemplateInjector",
]
