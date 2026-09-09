"""
Web scraper module for extracting and deduplicating images from web pages.
Includes SSRF protection with redirect validation, perceptual hashing (pHash/dHash),
and manifest generation.
"""

import ipaddress
import json
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from PIL import Image
import requests

from src.config import settings
from src.preprocess import load_image_safely
from src.utils import compute_sha256, logger


class SSRFSecurityError(ValueError):
    """Raised when a URL attempts to target internal, loopback, or private networks."""
    pass


def validate_url_security(url: str, denied_hosts: Optional[List[str]] = None) -> str:
    """
    Validates that a URL is safe to fetch (prevents SSRF attacks).
    - Checks HTTP / HTTPS schemes only.
    - Resolves hostname to IP and verifies it is not in private/loopback/reserved blocks.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFSecurityError(
            f"Prohibited URL scheme: '{parsed.scheme}'. Only http and https are allowed."
        )

    hostname = parsed.hostname
    if not hostname:
        raise SSRFSecurityError("Missing hostname in URL.")

    denied = denied_hosts or settings.denied_hosts
    if hostname.lower() in [h.lower() for h in denied]:
        raise SSRFSecurityError(f"Access to denied host '{hostname}' is blocked.")

    # Resolve IP address to prevent DNS rebinding or localhost escapes
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise SSRFSecurityError(f"DNS resolution failed for '{hostname}': {e}")

    for item in addr_info:
        ip_str = item[4][0]
        ip_obj = ipaddress.ip_address(ip_str)
        if (
            ip_obj.is_private
            or ip_obj.is_loopback
            or ip_obj.is_link_local
            or ip_obj.is_reserved
            or ip_obj.is_multicast
        ):
            raise SSRFSecurityError(
                f"SSRF violation: Host '{hostname}' resolved to prohibited IP '{ip_str}'."
            )

    return url


def safe_http_get(
    session: requests.Session,
    url: str,
    max_redirects: Optional[int] = None,
    timeout: int = 10,
    stream: bool = True,
) -> requests.Response:
    """
    Performs HTTP GET with manual redirect validation at each hop to prevent SSRF via open redirect.
    """
    max_redirs = max_redirects or settings.max_redirects
    current_url = url
    redirect_count = 0

    while True:
        validate_url_security(current_url)

        resp = session.get(
            current_url,
            timeout=timeout,
            stream=stream,
            allow_redirects=False,
        )

        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            redirect_count += 1
            if redirect_count > max_redirs:
                raise SSRFSecurityError(f"Exceeded maximum allowed redirects ({max_redirs}).")

            location = resp.headers.get("Location")
            if not location:
                raise SSRFSecurityError("Redirect missing Location header.")

            next_url = urljoin(current_url, location)
            logger.debug(f"Validating redirect hop {redirect_count}: {current_url} -> {next_url}")
            current_url = next_url
            continue

        return resp


def extract_candidate_image_urls(html_content: str, base_url: str) -> List[str]:
    """
    Parses HTML to find all candidate image URLs from <img> tags, data attributes, and <source> tags.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    found_urls: Set[str] = set()

    for img in soup.find_all(["img", "source"]):
        # Standard src
        src = img.get("src")
        if src and not src.startswith("data:"):
            found_urls.add(urljoin(base_url, src.strip()))

        # Lazy loading attributes
        for attr in ["data-src", "data-original", "data-lazy-src", "data-srcset"]:
            val = img.get(attr)
            if val and not val.startswith("data:"):
                first_url = val.strip().split(",")[0].split()[0]
                found_urls.add(urljoin(base_url, first_url))

        # srcset attribute
        srcset = img.get("srcset")
        if srcset:
            parts = [p.strip().split()[0] for p in srcset.split(",") if p.strip()]
            for part in parts:
                if not part.startswith("data:"):
                    found_urls.add(urljoin(base_url, part))

    return sorted(list(found_urls))


class WebpageImageScraper:
    """
    Extracts, validates, deduplicates, and saves images from any webpage.
    """

    def __init__(
        self,
        max_images: int = 50,
        timeout: int = 10,
        output_base_dir: Optional[Path] = None,
    ):
        self.max_images = max_images
        self.timeout = timeout
        self.output_base_dir = (output_base_dir or settings.temp_dir).resolve()
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.user_agent})

    def scrape_url(self, page_url: str) -> Tuple[List[Dict[str, Any]], Path]:
        """
        Scrapes all images from a webpage, deduplicates them using perceptual hashing,
        and saves a manifest.json.
        """
        logger.info(f"Scraping webpage: {page_url}")
        resp = safe_http_get(self.session, page_url, timeout=self.timeout, stream=True)
        resp.raise_for_status()

        # Read HTML with size cap (10MB max)
        content_chunks = []
        bytes_read = 0
        max_html_bytes = 10 * 1024 * 1024
        for chunk in resp.iter_content(chunk_size=65536):
            bytes_read += len(chunk)
            if bytes_read > max_html_bytes:
                raise ValueError("Webpage HTML exceeded maximum allowed size (10MB).")
            content_chunks.append(chunk)

        html_text = b"".join(content_chunks).decode(resp.encoding or "utf-8", errors="replace")
        candidate_urls = extract_candidate_image_urls(html_text, page_url)
        logger.info(f"Found {len(candidate_urls)} candidate image URLs.")

        # Create session directory
        session_id = compute_sha256(page_url.encode("utf-8"))[:12]
        session_dir = (self.output_base_dir / f"scrape_{session_id}").resolve()
        session_dir.mkdir(parents=True, exist_ok=True)

        manifest = self._download_and_deduplicate(candidate_urls, session_dir)

        # Write manifest.json
        manifest_path = session_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Completed scrape for {page_url}. Saved {len(manifest)} images to {session_dir}.")
        return manifest, session_dir

    def _download_and_deduplicate(
        self,
        urls: List[str],
        output_dir: Path,
    ) -> List[Dict[str, Any]]:
        """Downloads images, filters small/invalid images, and deduplicates using perceptual hash."""
        import imagehash

        manifest: List[Dict[str, Any]] = []
        known_hashes: List[Tuple[Any, str]] = []

        for url in urls[: self.max_images * 2]:
            if len(manifest) >= self.max_images:
                break

            try:
                img_resp = safe_http_get(self.session, url, timeout=self.timeout, stream=True)
                if img_resp.status_code != 200:
                    continue

                # Verify Content-Type matches image MIME
                content_type = img_resp.headers.get("Content-Type", "").lower().split(";")[0].strip()
                if content_type and not content_type.startswith("image/"):
                    logger.debug(f"Skipping non-image Content-Type '{content_type}' for {url}.")
                    continue

                # Stream image bytes with 15MB limit
                img_chunks = []
                img_bytes_count = 0
                for chunk in img_resp.iter_content(chunk_size=32768):
                    img_bytes_count += len(chunk)
                    if img_bytes_count > settings.max_image_bytes:
                        raise ValueError("Image file size exceeded 15MB limit.")
                    img_chunks.append(chunk)

                raw_bytes = b"".join(img_chunks)
                if len(raw_bytes) < 500:  # Skip tiny tracking pixels
                    continue

                # Safely parse and validate image
                pil_img = load_image_safely(
                    raw_bytes,
                    min_size=(settings.min_image_width, settings.min_image_height),
                )
                w, h = pil_img.size

                # Compute perceptual hashes
                phash = imagehash.phash(pil_img)
                dhash = imagehash.dhash(pil_img)

                # Check for duplicates (Hamming distance <= 2)
                is_duplicate = False
                for existing_hash, existing_file in known_hashes:
                    if phash - existing_hash <= 2:
                        is_duplicate = True
                        logger.debug(f"Skipping near-duplicate image ({url}) matching {existing_file}.")
                        break

                if is_duplicate:
                    continue

                # Unique image - save to disk safely
                sha256_val = compute_sha256(raw_bytes)
                file_name = f"img_{len(manifest):03d}_{sha256_val[:8]}.jpg"
                file_path = (output_dir / file_name).resolve()

                # Guard against path traversal
                if not file_path.is_relative_to(output_dir):
                    raise ValueError("Resolved file path escapes output directory.")

                pil_img.save(file_path, format="JPEG", quality=95)

                known_hashes.append((phash, file_name))
                manifest.append(
                    {
                        "index": len(manifest),
                        "url": url,
                        "local_path": str(file_path),
                        "filename": file_name,
                        "width": w,
                        "height": h,
                        "sha256": sha256_val,
                        "phash": str(phash),
                        "dhash": str(dhash),
                    }
                )

            except Exception as e:
                logger.debug(f"Failed to process image from {url}: {e}")
                continue

        return manifest
