"""Upload service for Cloudflare Images."""

import os
import uuid
from datetime import datetime

import httpx

from generationengine.models.errors import ErrorCode
from generationengine.services.retry_service import RetryableError


class UploadService:
    """Service for uploading images to Cloudflare Images API."""

    def __init__(
        self,
        account_id: str | None = None,
        api_token: str | None = None,
        public_url_base: str | None = None,
    ):
        """
        Initialize upload service.

        Args:
            account_id: Cloudflare account ID (defaults to CLOUDFLARE_ACCOUNT_ID env var)
            api_token: Cloudflare API token (defaults to CLOUDFLARE_IMAGES_API_TOKEN env var)
            public_url_base: Base URL for public image URLs (optional, auto-detected from API response)
        """
        self.account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        self.api_token = api_token or os.getenv("CLOUDFLARE_IMAGES_API_TOKEN")
        self.public_url_base = public_url_base

        if not self.account_id:
            raise ValueError("CLOUDFLARE_ACCOUNT_ID environment variable or account_id parameter is required")
        if not self.api_token:
            raise ValueError(
                "CLOUDFLARE_IMAGES_API_TOKEN environment variable or api_token parameter is required"
            )

        self.upload_url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/images/v1"

    async def upload_image(
        self,
        image_bytes: bytes,
        prefix: str = "generated",
        filename: str | None = None,
    ) -> str:
        """
        Upload image bytes to Cloudflare Images and return public URL.

        Args:
            image_bytes: Image data as bytes
            prefix: Prefix for generated filename (e.g., "generated", "statblock")
            filename: Optional specific filename (if not provided, generates one)

        Returns:
            Public URL of uploaded image

        Raises:
            RetryableError: For retryable failures (timeouts, rate limits)
            Exception: For non-retryable failures
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            filename = f"{prefix}_{timestamp}_{unique_id}.png"

        headers = {
            "Authorization": f"Bearer {self.api_token}",
        }

        files = {
            "file": (filename, image_bytes, "image/png"),
            "metadata": (None, '{"key":"value"}'),
            "requireSignedURLs": (None, "false"),
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.upload_url, headers=headers, files=files)

                if response.status_code == 429:
                    raise RetryableError(
                        ErrorCode.RATE_LIMITED,
                        "Cloudflare Images rate limit exceeded",
                    )

                if response.status_code != 200:
                    error_text = response.text
                    raise RetryableError(
                        ErrorCode.INTERNAL_ERROR,
                        f"Cloudflare Images API error {response.status_code}: {error_text}",
                    )

                result = response.json()["result"]
                public_url = result.get("variants", [None])[0]

                if not public_url:
                    raise RetryableError(
                        ErrorCode.INTERNAL_ERROR,
                        "Cloudflare Images API returned no public URL",
                    )

                # Ensure URL ends with /public
                if not public_url.endswith("/public"):
                    public_url = "/".join(public_url.split("/")[:-1]) + "/public"

                return public_url

        except httpx.TimeoutException as e:
            raise RetryableError(
                ErrorCode.PROVIDER_TIMEOUT,
                f"Cloudflare Images upload timed out: {str(e)}",
                original_exception=e,
            )
        except RetryableError:
            # Re-raise retryable errors as-is
            raise
        except Exception as e:
            raise RetryableError(
                ErrorCode.INTERNAL_ERROR,
                f"Cloudflare Images upload failed: {str(e)}",
                original_exception=e,
            )

