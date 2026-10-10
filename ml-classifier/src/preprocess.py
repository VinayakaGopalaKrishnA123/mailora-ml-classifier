"""
Text Preprocessing and Feature Normalization for Mailora Email Classifier.
"""

import re
import html
import unicodedata
from typing import Optional


# Precompiled regex patterns for speed
RE_HTML_TAGS = re.compile(r"<[^>]+>", re.IGNORECASE)
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
RE_EXTRA_SPACES = re.compile(r"\s+")
RE_BULLETS = re.compile(r"^[\s*•\-–—]+", re.MULTILINE)
RE_PUNCT_SPAM = re.compile(r"([!?,.:;])\1+")


def clean_text(text: Optional[str], preserve_urls: bool = False) -> str:
    """
    Cleans raw email text:
    - Decodes HTML entities
    - Strips HTML tags
    - Normalizes unicode characters
    - Cleans up repeated punctuation (e.g., '???' -> '?')
    - Optionally normalizes URLs and emails
    - Strips excessive whitespace
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Strip HTML tags
    text = RE_HTML_TAGS.sub(" ", text)

    # Normalize unicode (NFKD)
    text = unicodedata.normalize("NFKD", text)

    # Normalize repetitive punctuation
    text = RE_PUNCT_SPAM.sub(r"\1", text)

    # Normalize URLs & emails if requested
    if not preserve_urls:
        text = RE_URL.sub(" URL_LINK ", text)
        text = RE_EMAIL.sub(" EMAIL_ADDR ", text)

    # Normalize whitespace
    text = RE_EXTRA_SPACES.sub(" ", text).strip()
    return text


def truncate_smart(text: str, max_chars: int = 4000) -> str:
    """
    Preserves the most critical sections of long emails:
    - First 60% of max_chars (opening context, role details, status)
    - Last 40% of max_chars (application instructions, deadline, signature)
    """
    if len(text) <= max_chars:
        return text

    head_len = int(max_chars * 0.6)
    tail_len = max_chars - head_len
    return text[:head_len] + "\n...[TRUNCATED]...\n" + text[-tail_len:]


def combine_subject_and_body(
    subject: Optional[str],
    body: Optional[str],
    subject_weight: int = 2,
    max_body_chars: int = 4000,
) -> str:
    """
    Intelligently combines subject and body text for classification.
    Subject tokens carry high intent signal density, so repeating the subject
    gives appropriate statistical weight in bag-of-words / TF-IDF representations.
    """
    clean_subj = clean_text(subject)
    clean_b = truncate_smart(clean_text(body), max_chars=max_body_chars)

    if not clean_subj and not clean_b:
        return ""

    if not clean_subj:
        return clean_b

    if not clean_b:
        return clean_subj

    # Weight the subject by repeating or prefixing
    weighted_subj = " ".join([clean_subj] * max(1, subject_weight))
    return f"Subject: {clean_subj}. {weighted_subj} Body: {clean_b}"


def text_dedup_key(subject: Optional[str], body: Optional[str]) -> str:
    """
    Generates a normalized canonical key to identify duplicates
    or near-identical boilerplate emails to prevent leakage across splits.
    """
    s = clean_text(subject).lower()
    b = clean_text(body).lower()

    # Strip non-alphanumeric
    s_clean = re.sub(r"[^a-z0-9]", "", s)
    b_clean = re.sub(r"[^a-z0-9]", "", b[:300]) # First 300 chars of body
    return f"{s_clean}___{b_clean}"
