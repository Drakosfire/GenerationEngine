"""Tests for text generation service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from generationengine.models.requests import TextGenerationRequest, TextModel
from generationengine.models.text_responses import TextGenerationResponse
from generationengine.models.errors import ErrorCode
from generationengine.services.text_service import TextGenerationService


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "Generated text content"
    response.choices[0].message.refusal = None  # Explicitly set to None for structured outputs check
    response.usage = MagicMock()
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    response.usage.total_tokens = 30
    return response


@pytest.fixture
def text_service():
    """Create a TextGenerationService with mocked OpenAI client."""
    with patch("generationengine.services.text_service.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        service = TextGenerationService(openai_api_key="test-key")
        service.openai_client = mock_client
        return service


@pytest.mark.asyncio
async def test_text_generation_success(text_service, mock_openai_response):
    """Test successful text generation."""
    text_service.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    request = TextGenerationRequest(
        user_prompt="Generate a description",
        model=TextModel.GPT_5_1,
    )

    response = await text_service.generate(request)

    assert response.success is True
    assert response.content == "Generated text content"
    assert response.error is None
    assert response.metrics is not None
    assert response.metrics.tokens_used == 30
    assert response.metrics.model_used == "gpt-5.1"
    assert response.metrics.retry_count == 0


@pytest.mark.asyncio
async def test_text_generation_with_system_prompt(text_service, mock_openai_response):
    """Test text generation with system prompt."""
    text_service.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    request = TextGenerationRequest(
        system_prompt="You are a helpful assistant",
        user_prompt="Generate a description",
        model=TextModel.GPT_5_1,
    )

    response = await text_service.generate(request)

    assert response.success is True
    assert response.content == "Generated text content"
    
    # Verify OpenAI was called with system prompt
    call_args = text_service.openai_client.chat.completions.create.call_args
    messages = call_args[1]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant"
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Generate a description"


@pytest.mark.asyncio
async def test_text_generation_with_parameters(text_service, mock_openai_response):
    """Test text generation with custom parameters."""
    text_service.openai_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)

    request = TextGenerationRequest(
        user_prompt="Generate text",
        model=TextModel.GPT_5_1,
        temperature=0.9,
        max_tokens=1000,
    )

    response = await text_service.generate(request)

    assert response.success is True
    
    # Verify OpenAI was called with correct parameters
    call_args = text_service.openai_client.chat.completions.create.call_args
    assert call_args[1]["model"] == "gpt-5.1"
    assert call_args[1]["temperature"] == 0.9
    assert call_args[1]["max_tokens"] == 1000


@pytest.mark.asyncio
async def test_text_generation_rate_limit_error(text_service):
    """Test handling of rate limit error."""
    request = TextGenerationRequest(user_prompt="Test prompt")

    # Mock retry_with_backoff to raise RetryableError after retries exhausted
    with patch("generationengine.services.text_service.retry_with_backoff") as mock_retry:
        from generationengine.services.retry_service import RetryableError
        mock_retry.side_effect = RetryableError(
            ErrorCode.RATE_LIMITED,
            "OpenAI API rate limit exceeded: Rate limit exceeded",
        )

        response = await text_service.generate(request)

        assert response.success is False
        assert response.error is not None
        assert response.error.code == ErrorCode.RATE_LIMITED
        assert response.error.retryable is True
        assert response.content is None


@pytest.mark.asyncio
async def test_text_generation_timeout_error(text_service):
    """Test handling of timeout error."""
    request = TextGenerationRequest(user_prompt="Test prompt")

    # Mock retry_with_backoff to raise RetryableError after retries exhausted
    with patch("generationengine.services.text_service.retry_with_backoff") as mock_retry:
        from generationengine.services.retry_service import RetryableError
        mock_retry.side_effect = RetryableError(
            ErrorCode.PROVIDER_TIMEOUT,
            "OpenAI API request timed out: Request timed out",
        )

        response = await text_service.generate(request)

        assert response.success is False
        assert response.error is not None
        assert response.error.code == ErrorCode.PROVIDER_TIMEOUT
        assert response.error.retryable is True


@pytest.mark.asyncio
async def test_text_generation_service_initialization_no_key():
    """Test that service initialization fails without API key."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            TextGenerationService()


@pytest.mark.asyncio
async def test_text_generation_service_initialization_with_key():
    """Test that service initialization succeeds with API key."""
    with patch("generationengine.services.text_service.AsyncOpenAI") as mock_openai:
        service = TextGenerationService(openai_api_key="test-key")
        assert service.openai_client is not None
        mock_openai.assert_called_once_with(api_key="test-key")


def test_text_generation_request_validation():
    """Test TextGenerationRequest model validation."""
    request = TextGenerationRequest(
        user_prompt="Test prompt",
        model=TextModel.GPT_5_1,
        temperature=0.7,
        max_tokens=1000,
    )

    assert request.user_prompt == "Test prompt"
    assert request.model == TextModel.GPT_5_1
    assert request.temperature == 0.7
    assert request.max_tokens == 1000
    assert request.system_prompt is None


def test_text_generation_request_defaults():
    """Test TextGenerationRequest default values."""
    request = TextGenerationRequest(user_prompt="Test prompt")

    assert request.model == TextModel.GPT_5_1
    assert request.temperature == 0.7
    assert request.max_tokens is None
    assert request.system_prompt is None


def test_text_generation_request_user_prompt_required():
    """Test that user_prompt is required."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        TextGenerationRequest()


def test_text_generation_request_temperature_range():
    """Test that temperature is constrained to 0.0-2.0."""
    # Valid ranges
    TextGenerationRequest(user_prompt="test", temperature=0.0)
    TextGenerationRequest(user_prompt="test", temperature=2.0)

    # Invalid ranges
    with pytest.raises(Exception):  # Pydantic ValidationError
        TextGenerationRequest(user_prompt="test", temperature=-0.1)

    with pytest.raises(Exception):  # Pydantic ValidationError
        TextGenerationRequest(user_prompt="test", temperature=2.1)

