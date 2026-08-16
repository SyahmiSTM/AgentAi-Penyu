"""Improved web scraper Lambda for extracting web content and answering questions."""
import json
import re
import urllib.request
import urllib.error
import urllib.parse
from html import unescape
from html.parser import HTMLParser
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_RETRIES = 2
RETRY_DELAY = 1.0
REQUEST_TIMEOUT = 10
# NOTE: this excerpt is INPUT to the agent, and the token bonus is scored on the
# agent's OUTPUT tokens only -- so shrinking this does NOT increase the score.
# It is kept well below the original 4000 because a smaller excerpt is faster (the
# run is clock-limited) and gives the model less irrelevant text to ramble about,
# but it is deliberately not squeezed to the minimum: cutting off the answer costs
# 800 points, and there is no token bonus to win back in exchange.
MAX_RESPONSE_LENGTH = 1200
# Candidate sentences considered when filling the excerpt budget, best-scoring first.
TOP_SENTENCES = 6
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# HTML Parser - extracts structured content from HTML
# ---------------------------------------------------------------------------
class ContentExtractor(HTMLParser):
    """Extracts structured text content from HTML, preserving semantic structure."""

    SKIP_TAGS = frozenset([
        "script", "style", "noscript", "iframe", "svg", "math",
        "nav", "footer", "header", "aside",
    ])
    BLOCK_TAGS = frozenset([
        "p", "div", "section", "article", "main", "h1", "h2", "h3",
        "h4", "h5", "h6", "li", "tr", "blockquote", "pre", "br", "hr",
    ])
    HEADING_TAGS = frozenset(["h1", "h2", "h3", "h4", "h5", "h6"])

    def __init__(self):
        super().__init__()
        self.title = ""
        self.headings = []
        self.links = []
        self.paragraphs = []
        self._current_text = []
        self._skip_depth = 0
        self._in_title = False
        self._in_heading = False
        self._current_heading = ""
        self._current_tag_stack = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        self._current_tag_stack.append(tag)

        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
            return

        if self._skip_depth > 0:
            return

        if tag == "title":
            self._in_title = True
        elif tag in self.HEADING_TAGS:
            self._in_heading = True
            self._current_heading = ""
        elif tag == "a":
            attrs_dict = dict(attrs)
            href = attrs_dict.get("href", "")
            if href and not href.startswith(("#", "javascript:")):
                self.links.append(href)

        if tag in self.BLOCK_TAGS:
            self._current_text.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if self._current_tag_stack and self._current_tag_stack[-1] == tag:
            self._current_tag_stack.pop()

        if tag in self.SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return

        if self._skip_depth > 0:
            return

        if tag == "title":
            self._in_title = False
        elif tag in self.HEADING_TAGS and self._in_heading:
            self._in_heading = False
            heading_text = self._current_heading.strip()
            if heading_text:
                self.headings.append(heading_text)

        if tag in self.BLOCK_TAGS:
            self._current_text.append("\n")

    def handle_data(self, data):
        if self._skip_depth > 0:
            return

        text = data.strip()
        if not text:
            return

        if self._in_title:
            self.title += text
        if self._in_heading:
            self._current_heading += text

        self._current_text.append(text)

    def handle_entityref(self, name):
        char = unescape(f"&{name};")
        self.handle_data(char)

    def handle_charref(self, name):
        char = unescape(f"&#{name};")
        self.handle_data(char)

    def get_text(self):
        """Return cleaned text with normalized whitespace."""
        raw = " ".join(self._current_text)
        # Normalize whitespace: collapse multiple spaces, preserve newlines
        lines = raw.split("\n")
        cleaned = []
        for line in lines:
            line = re.sub(r"[ \t]+", " ", line).strip()
            if line:
                cleaned.append(line)
        return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# URL Extraction and Validation
# ---------------------------------------------------------------------------
def extract_url(text):
    """Extract and validate URL from text, handling edge cases."""
    # Match URLs with common schemes
    url_pattern = r'https?://[^\s<>"\'`\]\[{}|\\^)]*[^\s<>"\'`\]\[{}|\\^.,;:!?)\'\"]'
    matches = re.findall(url_pattern, text)

    if not matches:
        # Try a more lenient pattern
        matches = re.findall(r'https?://[^\s]+', text)

    if not matches:
        return None

    url = matches[0]
    # Clean trailing punctuation that's likely not part of the URL
    url = re.sub(r'[.,;:!?)"\'>]+$', '', url)

    # Validate URL structure
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return None
        # Basic domain validation
        if '.' not in parsed.netloc and 'localhost' not in parsed.netloc:
            return None
    except Exception:
        return None

    return url


# ---------------------------------------------------------------------------
# HTTP Fetching with Retries
# ---------------------------------------------------------------------------
def fetch_url(url, retries=MAX_RETRIES, timeout=REQUEST_TIMEOUT):
    """Fetch URL content with retries and proper error handling."""
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity",
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                charset = _detect_charset(content_type)

                raw_bytes = resp.read()
                # Try declared charset first, fall back to utf-8, then latin-1
                content = _decode_content(raw_bytes, charset)

                return {
                    "content": content,
                    "content_type": content_type,
                    "url": resp.url,  # Final URL after redirects
                    "status": resp.status,
                }
        except urllib.error.HTTPError as e:
            last_error = f"HTTP {e.code}: {e.reason}"
            # Don't retry client errors (4xx)
            if 400 <= e.code < 500:
                break
        except urllib.error.URLError as e:
            last_error = f"URL Error: {e.reason}"
        except TimeoutError:
            last_error = "Request timed out"
        except Exception as e:
            last_error = f"Fetch error: {type(e).__name__}: {str(e)}"

        if attempt < retries:
            time.sleep(RETRY_DELAY * (attempt + 1))

    return {"error": last_error}


def _detect_charset(content_type):
    """Extract charset from Content-Type header."""
    if not content_type:
        return None
    match = re.search(r'charset=([^\s;]+)', content_type, re.IGNORECASE)
    return match.group(1).strip('"\'') if match else None


def _decode_content(raw_bytes, charset=None):
    """Decode bytes to string, trying multiple encodings."""
    encodings = []
    if charset:
        encodings.append(charset)
    encodings.extend(["utf-8", "latin-1", "cp1252"])

    for encoding in encodings:
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    # Last resort
    return raw_bytes.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Content Relevance Extraction
# ---------------------------------------------------------------------------
def extract_relevant_content(full_text, question, max_length=MAX_RESPONSE_LENGTH):
    """Extract the most relevant portions of text based on the question."""
    if not question or len(full_text) <= max_length:
        return full_text[:max_length]

    # Extract meaningful keywords from the question (remove URLs and stop words)
    q_clean = re.sub(r'https?://[^\s]+', '', question).lower()
    stop_words = frozenset([
        "what", "which", "where", "when", "who", "how", "why", "is", "are",
        "was", "were", "the", "a", "an", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "this", "that", "it", "its", "can",
        "do", "does", "did", "will", "would", "could", "should", "may",
        "might", "according", "about", "please", "tell", "me", "find",
        "give", "get", "show", "look", "up", "and", "or", "but", "not",
    ])
    keywords = [
        w for w in re.findall(r'\b[a-z]{2,}\b', q_clean)
        if w not in stop_words
    ]

    if not keywords:
        return full_text[:max_length]

    # Score sentences by keyword relevance
    sentences = re.split(r'(?<=[.!?])\s+|\n', full_text)
    scored = []
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        score = sum(
            3 if kw in sentence_lower else 0
            for kw in keywords
        )
        # Bonus for exact multi-word matches
        for j in range(len(keywords) - 1):
            bigram = f"{keywords[j]} {keywords[j+1]}"
            if bigram in sentence_lower:
                score += 5
        # Position bonus (content near the top is often more relevant)
        position_bonus = max(0, 1.0 - (i / max(len(sentences), 1)) * 0.3)
        score *= position_bonus
        if score > 0:
            scored.append((score, i, sentence))

    if not scored:
        return full_text[:max_length]

    # Sort by score descending, then by position for ties
    scored.sort(key=lambda x: (-x[0], x[1]))

    # Fill the budget in SCORE order, so the best-matching sentence can never be
    # crowded out by an earlier-positioned but weaker one -- then restore reading
    # order for the returned string. Uses `continue` rather than `break` so a single
    # long sentence does not block shorter high-scoring ones behind it.
    chosen = []
    current_length = 0
    for _, idx, sentence in scored[:TOP_SENTENCES]:
        if current_length + len(sentence) + 1 > max_length:
            continue
        chosen.append((idx, sentence))
        current_length += len(sentence) + 1

    if not chosen:
        # Every candidate individually exceeds the budget: return the single
        # best-scoring sentence truncated, rather than the top of the page.
        return scored[0][2][:max_length]

    chosen.sort(key=lambda x: x[0])
    return " ".join(sentence for _, sentence in chosen)


# ---------------------------------------------------------------------------
# Main Handler
# ---------------------------------------------------------------------------
def lambda_handler(event, context):
    """Lambda handler for web scraping requests."""
    try:
        # Parse input - handle API Gateway and direct invocation
        body = _parse_event(event)

        question = (body.get("question") or body.get("message") or "").strip()
        if not question:
            return _response(error="No question or message provided")

        # Extract URL
        url = extract_url(question)
        if not url:
            return _response(error="No valid URL found in the input")

        # Fetch the page
        result = fetch_url(url)
        if "error" in result:
            return _response(error=result["error"], url=url)

        content_type = result.get("content_type", "")
        content = result["content"]
        final_url = result.get("url", url)

        # Handle non-HTML content
        if "html" not in content_type and "text" not in content_type:
            # For JSON responses, return directly
            if "json" in content_type:
                try:
                    parsed_json = json.loads(content)
                    return _response(
                        answer=json.dumps(parsed_json, indent=2)[:MAX_RESPONSE_LENGTH],
                        url=final_url,
                        content_type="json",
                    )
                except json.JSONDecodeError:
                    pass
            # For plain text
            return _response(
                answer=content[:MAX_RESPONSE_LENGTH],
                url=final_url,
                content_type="text",
            )

        # Parse HTML
        extractor = ContentExtractor()
        try:
            extractor.feed(content)
        except Exception:
            # Fallback: regex-based stripping if parser fails
            text = _fallback_strip_html(content)
            return _response(
                answer=text[:MAX_RESPONSE_LENGTH],
                url=final_url,
                content_type="html_fallback",
            )

        full_text = extractor.get_text()

        # Extract relevant content based on the question
        relevant_text = extract_relevant_content(full_text, question)

        # Title and headings are deliberately NOT returned: they cost input tokens on
        # every web challenge and the answer is always in the excerpt, not the nav
        # structure. extractor.title/.headings remain available for debugging.
        return _response(
            answer=relevant_text,
            url=final_url,
            content_type="html",
        )

    except Exception as e:
        return _response(error=f"Unexpected error: {type(e).__name__}: {str(e)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_event(event):
    """Parse the Lambda event from various invocation sources."""
    if isinstance(event, dict) and "body" in event:
        body = event["body"]
        if isinstance(body, str):
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"question": body}
        return body if isinstance(body, dict) else {}
    return event if isinstance(event, dict) else {}


def _response(answer=None, error=None, url=None, content_type=None):
    """Build a standardized response.

    Kept intentionally minimal: every field here becomes input tokens for the agent.
    """
    body = {"success": error is None}
    if answer:
        body["answer"] = answer
    if error:
        body["error"] = error
    if url:
        body["url"] = url
    if content_type:
        body["content_type"] = content_type
    return {
        "statusCode": 200,
        "body": json.dumps(body),
    }


def _fallback_strip_html(html):
    """Regex-based HTML stripping as fallback."""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
