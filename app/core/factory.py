"""Component Factory for strategy instantiation.

The Factory Pattern allows the application to instantiate
different strategy implementations at runtime based on
configuration or environment variables.
"""

import logging

from app.core.config import Settings, get_settings
from app.interfaces.template import BaseTemplateAnalyzer, BaseTemplateInjector
from app.strategies.template_engine import TemplateAnalyzer, TemplateInjector

logger = logging.getLogger(__name__)


class ComponentFactory:
    """Factory for creating component instances based on configuration.

    This factory implements the Factory Pattern, allowing strategy
    selection at runtime without modifying core application code.

    Example:
        ```python
        settings = get_settings()
        factory = ComponentFactory(settings)

        analyzer = factory.get_template_analyzer()
        injector = factory.get_template_injector()
        ```
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the factory with optional settings.

        Args:
            settings: Application settings. If None, uses global settings.
        """
        self._settings = settings or get_settings()
        self._template_analyzer_cache: BaseTemplateAnalyzer | None = None
        self._template_injector_cache: BaseTemplateInjector | None = None

    def get_template_analyzer(self, custom_prompt: str | None = None) -> BaseTemplateAnalyzer:
        """Get a template analyzer instance.

        Args:
            custom_prompt: Optional custom prompt for LLM analysis.
                When provided, bypasses cache to ensure prompt isolation.

        Returns:
            A BaseTemplateAnalyzer implementation instance.

        Raises:
            ValueError: If template analyzer configuration is invalid.
        """
        # Bypass cache when custom prompt is provided for prompt isolation
        if custom_prompt is not None:
            logger.info("Instantiating template analyzer with custom prompt")
            return TemplateAnalyzer(
                use_llm=self._settings.use_llm_for_templates,
                openai_api_key=self._settings.openrouter_api_key,
                base_url=self._settings.openai_base_url,
                model=self._settings.llm_chat_model,
                custom_prompt=custom_prompt,
            )

        if self._template_analyzer_cache is None:
            logger.info("Instantiating template analyzer")

            self._template_analyzer_cache = TemplateAnalyzer(
                use_llm=self._settings.use_llm_for_templates,
                openai_api_key=self._settings.openrouter_api_key,
                base_url=self._settings.openai_base_url,
                model=self._settings.llm_chat_model,
            )

        return self._template_analyzer_cache

    def get_template_injector(self) -> BaseTemplateInjector:
        """Get a template injector instance.

        Returns:
            A BaseTemplateInjector implementation instance.
        """
        if self._template_injector_cache is None:
            logger.info("Instantiating template injector")

            self._template_injector_cache = TemplateInjector()

        return self._template_injector_cache

    def clear_cache(self) -> None:
        """Clear all cached component instances.

        This forces new instances to be created on next access.
        Useful for testing or when settings change.
        """
        self._template_analyzer_cache = None
        self._template_injector_cache = None
        logger.debug("Component factory cache cleared")


# Global factory instance
_factory: ComponentFactory | None = None


def get_factory() -> ComponentFactory:
    """Get or create the global ComponentFactory instance.

    Returns:
        The singleton ComponentFactory instance.
    """
    global _factory
    if _factory is None:
        _factory = ComponentFactory()
    return _factory
