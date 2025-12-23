"""Tests for text generation service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from generationengine.models.requests import TextGenerationRequest, TextModel
from generationengine.models.text_responses import TextGenerationResponse
from generationengine.models.errors import ErrorCode
from generationengine.services.text_service import TextGenerationService


@pytest.fixture
def mock_openai_response():
    """Create a mock OpenAI Responses API response."""
    response = MagicMock()
    response.output_text = "Generated text content"
    response.refusal = None  # Explicitly set to None for structured outputs check
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
    text_service.openai_client.responses.create = AsyncMock(return_value=mock_openai_response)

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
    text_service.openai_client.responses.create = AsyncMock(return_value=mock_openai_response)

    request = TextGenerationRequest(
        system_prompt="You are a helpful assistant",
        user_prompt="Generate a description",
        model=TextModel.GPT_5_1,
    )

    response = await text_service.generate(request)

    assert response.success is True
    assert response.content == "Generated text content"
    
    # Verify OpenAI was called with instructions (system prompt) and input
    call_args = text_service.openai_client.responses.create.call_args
    call_kwargs = call_args.kwargs if call_args else {}
    assert call_kwargs.get("instructions") == "You are a helpful assistant"
    assert call_kwargs.get("input") == "Generate a description"


@pytest.mark.asyncio
async def test_text_generation_with_parameters(text_service, mock_openai_response):
    """Test text generation with custom parameters."""
    text_service.openai_client.responses.create = AsyncMock(return_value=mock_openai_response)

    request = TextGenerationRequest(
        user_prompt="Generate text",
        model=TextModel.GPT_5_1,
        temperature=0.9,
        max_tokens=1000,
    )

    response = await text_service.generate(request)

    assert response.success is True
    
    # Verify OpenAI was called with correct parameters
    call_args = text_service.openai_client.responses.create.call_args
    call_kwargs = call_args.kwargs if call_args else {}
    assert call_kwargs.get("model") == "gpt-5.1"
    assert call_kwargs.get("temperature") == 0.9
    assert call_kwargs.get("max_tokens") == 1000


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


@pytest.mark.asyncio
async def test_text_generation_stream_success(text_service):
    """Test successful text generation streaming."""
    # Mock the streaming response
    mock_event1 = MagicMock()
    mock_event1.type = "response.output_text.delta"
    mock_event1.delta = "Hello"
    
    mock_event2 = MagicMock()
    mock_event2.type = "response.output_text.delta"
    mock_event2.delta = " world"
    
    mock_event3 = MagicMock()
    mock_event3.type = "response.completed"
    
    # Create async generator for events
    async def mock_stream():
        yield mock_event1
        yield mock_event2
        yield mock_event3
    
    # Mock the stream manager context
    mock_stream_manager = MagicMock()
    mock_stream_manager.__aenter__ = AsyncMock(return_value=mock_stream())
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)
    
    text_service.openai_client.responses.stream = MagicMock(return_value=mock_stream_manager)
    
    request = TextGenerationRequest(
        user_prompt="Generate a greeting",
        model=TextModel.GPT_5_1,
    )
    
    chunks = []
    async for chunk in text_service.generate_stream(request):
        chunks.append(chunk)
    
    assert len(chunks) == 3  # Two content chunks + [DONE]
    assert "data: Hello\n\n" in chunks
    assert "data:  world\n\n" in chunks
    assert "data: [DONE]\n\n" in chunks
    
    # Verify OpenAI was called with correct parameters
    call_args = text_service.openai_client.responses.stream.call_args
    call_kwargs = call_args.kwargs if call_args else {}
    assert call_kwargs.get("model") == "gpt-5.1"
    assert call_kwargs.get("input") == "Generate a greeting"


@pytest.mark.asyncio
async def test_text_generation_stream_with_system_prompt(text_service):
    """Test text generation streaming with system prompt."""
    mock_event = MagicMock()
    mock_event.type = "response.output_text.delta"
    mock_event.delta = "Response"
    
    mock_completed = MagicMock()
    mock_completed.type = "response.completed"
    
    async def mock_stream():
        yield mock_event
        yield mock_completed
    
    mock_stream_manager = MagicMock()
    mock_stream_manager.__aenter__ = AsyncMock(return_value=mock_stream())
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)
    
    text_service.openai_client.responses.stream = MagicMock(return_value=mock_stream_manager)
    
    request = TextGenerationRequest(
        system_prompt="You are a helpful assistant",
        user_prompt="Generate text",
        model=TextModel.GPT_5_1,
    )
    
    chunks = []
    async for chunk in text_service.generate_stream(request):
        chunks.append(chunk)
    
    # Verify OpenAI was called with instructions
    call_args = text_service.openai_client.responses.stream.call_args
    call_kwargs = call_args.kwargs if call_args else {}
    assert call_kwargs.get("instructions") == "You are a helpful assistant"
    assert call_kwargs.get("input") == "Generate text"


@pytest.mark.asyncio
async def test_text_generation_stream_error_event(text_service):
    """Test handling of error event in streaming."""
    mock_error = MagicMock()
    mock_error.type = "response.error"
    mock_error.error = MagicMock()
    mock_error.error.message = "Test error"
    
    async def mock_stream():
        yield mock_error
    
    mock_stream_manager = MagicMock()
    mock_stream_manager.__aenter__ = AsyncMock(return_value=mock_stream())
    mock_stream_manager.__aexit__ = AsyncMock(return_value=None)
    
    text_service.openai_client.responses.stream = MagicMock(return_value=mock_stream_manager)
    
    request = TextGenerationRequest(user_prompt="Test")
    
    chunks = []
    async for chunk in text_service.generate_stream(request):
        chunks.append(chunk)
    
    assert len(chunks) == 1
    assert "[ERROR]Test error" in chunks[0]

