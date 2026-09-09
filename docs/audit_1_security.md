# Security Audit 1: Application, Network & Input Security

**Date**: 2026-09-09  
**Target**: AI Image Detection MLOps Pipeline (`src/`)  
**Auditor**: Antigravity Automated Security Review  
**Tooling**: Bandit v1.9.4 AST Analyzer, Code Inspection, SSRF Penetration Assessment

---

## 1. Executive Summary
A comprehensive security review was conducted on the core components of the AI Image Detection MLOps pipeline. The review targeted network attack surfaces (web image scraper), file ingestion boundaries (image decompression & decoding), API endpoints, and dependency supply-chain security.

Four initial medium-severity issues were detected by static AST analysis (Bandit), alongside a critical design-level vulnerability in HTTP redirect handling for SSRF protection. All issues have been identified, remediated, and verified with automated test suites.

---

## 2. Findings and Vulnerabilities Identified

### [VULN-01] SSRF Bypass via HTTP Open Redirects (High Severity)
- **Component**: `src/scrape.py`
- **Description**: The scraper validated the initial URL against private IP blocks and loopback interfaces, but invoked `requests.get` with default redirect following (`allow_redirects=True`). An external website could return an HTTP 301/302 redirect header pointing to internal cloud metadata (`http://169.254.169.254/`) or internal cluster endpoints (`http://127.0.0.1:8000/`), completely bypassing initial perimeter validation.
- **Remediation**: Implemented manual iterative redirect resolution (`allow_redirects=False`) with a strict maximum hop count (5 hops). Every redirect destination URL and resolved IP address is verified through `validate_url_security()` before the request is issued.

### [VULN-02] Supply Chain Integrity: Unpinned Hugging Face Model Downloads (Medium Severity - CWE-494 / Bandit B615)
- **Component**: `src/model.py`, `src/dataset.py`, `src/config.py`
- **Description**: `from_pretrained` and `load_dataset` downloaded artifacts from the Hugging Face Hub without an explicit revision commit SHA or tag, creating vulnerability to upstream model tampering or compromised repository commits.
- **Remediation**: Added configurable `clip_revision` pinned to a verified immutable commit tag/revision in `Settings`, and passed `revision=settings.clip_revision` across all Hugging Face loaders.

### [VULN-03] Bandit False Positive B104 on Blacklist IP String (Low Severity - Bandit B104)
- **Component**: `src/config.py`
- **Description**: Bandit flagged `"0.0.0.0"` in `denied_hosts` as a potential hardcoded bind to all interfaces.
- **Remediation**: Clarified the blacklist configuration with inline `# nosec B104` documentation verifying it is used solely as a filter against inbound requests.

### [VULN-04] Image Decompression Bomb & Malicious MIME Flooding (Medium Severity - CWE-400)
- **Component**: `src/scrape.py`, `src/preprocess.py`
- **Description**: Arbitrary web responses with non-image payloads (e.g. huge text or binary blobs) could be buffered into memory or submitted to PIL, risking denial of service.
- **Remediation**: Enforced early `Content-Type` header verification against allowed image MIME types (`image/jpeg`, `image/png`, `image/webp`, `image/gif`), streamed downloads with strict byte counters, and enforced `Image.MAX_IMAGE_PIXELS = 50_000_000`.

### [VULN-05] Path Traversal and Local Directory Escapes (Low Severity - CWE-22)
- **Component**: `src/scrape.py`
- **Description**: File writes for scraped images must ensure the resolved file paths never escape the session output directory.
- **Remediation**: Standardized file naming to SHA256-derived hashes, verified output paths are strictly within the designated temporary directory, and sanitized all user inputs.

---

## 3. Verification
All remediations were tested with automated test cases covering loopback redirects, private IP blocking, non-image rejection, and clean Bandit scans.
