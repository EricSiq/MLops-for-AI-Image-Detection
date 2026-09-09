"""
Unit tests for web scraping, SSRF validation, and perceptual deduplication.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.scrape import (
    SSRFSecurityError,
    WebpageImageScraper,
    extract_candidate_image_urls,
    safe_http_get,
    validate_url_security,
)


def test_validate_url_security_blocks_private_ips():
    """Verify SSRF protection blocks localhost and private IP addresses."""
    with pytest.raises(SSRFSecurityError, match="Access to denied host"):
        validate_url_security("http://localhost/test")

    with pytest.raises(SSRFSecurityError, match="Access to denied host"):
        validate_url_security("http://127.0.0.1/admin")


def test_validate_url_security_blocks_bad_schemes():
    """Verify non-HTTP schemes (e.g. file://, gopher://, ftp://) are rejected."""
    with pytest.raises(SSRFSecurityError, match="Prohibited URL scheme"):
        validate_url_security("file:///etc/passwd")

    with pytest.raises(SSRFSecurityError, match="Prohibited URL scheme"):
        validate_url_security("ftp://example.com/file.jpg")


def test_safe_http_get_blocks_open_redirect_to_private_ip():
    """Verify that HTTP redirects to private IPs or metadata endpoints are blocked."""
    mock_session = MagicMock()
    # First response redirects to 127.0.0.1
    resp_redirect = MagicMock()
    resp_redirect.is_redirect = True
    resp_redirect.status_code = 302
    resp_redirect.headers = {"Location": "http://127.0.0.1:8000/internal"}
    mock_session.get.return_value = resp_redirect

    with patch("src.scrape.validate_url_security") as mock_val:
        # Allow first call, raise on second
        def side_effect(url):
            if "127.0.0.1" in url:
                raise SSRFSecurityError("Blocked private IP")
            return url

        mock_val.side_effect = side_effect

        with pytest.raises(SSRFSecurityError, match="Blocked private IP"):
            safe_http_get(mock_session, "https://public-site.com/redirect", max_redirects=3)


def test_extract_candidate_image_urls():
    """Verify HTML parsing captures standard src, data-src, and srcset."""
    html = """
    <html>
      <body>
        <img src="/images/banner.png" alt="Banner">
        <img data-src="https://cdn.example.com/pic1.jpg" alt="Lazy">
        <picture>
          <source srcset="/photos/photo1.webp 1x, /photos/photo1_2x.webp 2x">
          <img src="/photos/fallback.jpg">
        </picture>
        <img src="data:image/png;base64,iVBORw0KGgo..." alt="Inline Base64">
      </body>
    </html>
    """
    base = "https://example.com/blog/"
    urls = extract_candidate_image_urls(html, base)

    # Must resolve relative paths
    assert "https://example.com/images/banner.png" in urls
    assert "https://cdn.example.com/pic1.jpg" in urls
    assert "https://example.com/photos/photo1.webp" in urls
    assert "https://example.com/photos/fallback.jpg" in urls
    # Data URIs must be excluded
    assert not any(u.startswith("data:") for u in urls)
