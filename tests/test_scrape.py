"""
Unit tests for web scraping, SSRF validation, and perceptual deduplication.
"""

import pytest

from src.scrape import (
    SSRFSecurityError,
    extract_candidate_image_urls,
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
