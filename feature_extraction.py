# feature_extraction.py
import re
from urllib.parse import urlparse
import unicodedata
import math
import validators

# Trusted domains and brands
TRUSTED_DOMAINS = {
    "google.com", "amazon.in", "microsoft.com",
    "facebook.com", "apple.com", "github.com", "paypal.com"
}
BRANDS = ["google", "amazon", "microsoft", "facebook", "apple", "github", "paypal"]

# -------------------- Utility functions -------------------- #
def levenshtein(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0]*len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1] + (0 if ca==cb else 1))
        prev = cur
    return prev[-1]

def normalized_distance(a, b):
    a = a or ""
    b = b or ""
    return levenshtein(a,b)/max(1, max(len(a), len(b)))

def shannon_entropy(s):
    if not s: return 0.0
    freq = {}
    for ch in s: freq[ch] = freq.get(ch,0)+1
    probs = [v/len(s) for v in freq.values()]
    return -sum(p*math.log2(p) for p in probs)

def extract_sld(hostname):
    if not hostname: return ""
    parts = hostname.split('.')
    if len(parts)>=2: return parts[-2]
    return parts[0]

def is_punycode(hostname):
    return hostname.startswith("xn--") or "xn--" in hostname

# -------------------- Feature extraction -------------------- #
class FeatureExtraction:
    def __init__(self, url: str):
        self.raw = (url or "").strip()
        # normalize scheme
        if self.raw and not self.raw.startswith(("http://","https://")):
            self.norm = "http://" + self.raw
        else:
            self.norm = self.raw
        try:
            self.parsed = urlparse(self.norm)
        except Exception:
            self.parsed = None
        self.hostname = (self.parsed.hostname or "").lower() if self.parsed and self.parsed.hostname else ""
        self.path = self.parsed.path if self.parsed else ""
        self.query = self.parsed.query if self.parsed else ""

    def to_dict(self):
        u = self.norm or ""
        host = self.hostname or ""
        sld = extract_sld(host)

        # Detect suspicious '@' tricks
        at_trick = int('@' in u and (self.hostname == "" or bool(self.parsed.username)))
        malformed_hostname = int(self.hostname == "")
        suspicious_at = int(at_trick or malformed_hostname)

        feats = {
            # URL lexical
            "url_length": len(u),
            "path_length": len(self.path),
            "query_length": len(self.query),
            "num_dots": u.count('.'),
            "num_hyphens": u.count('-'),
            "num_underscores": u.count('_'),
            "num_at": u.count('@'),
            "num_question": u.count('?'),
            "num_eq": u.count('='),
            "num_percent": u.count('%'),
            "num_digits": sum(ch.isdigit() for ch in u),
            "num_letters": sum(ch.isalpha() for ch in u),
            # Hostname
            "hostname_length": len(host),
            "sld_length": len(sld),
            "num_subdomains": max(len(host.split('.'))-2,0) if host else 0,
            "has_ip": int(bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host))),
            "is_punycode": int(is_punycode(host)),
            # Entropy
            "hostname_entropy": shannon_entropy(host),
            "path_entropy": shannon_entropy(self.path),
            # Brand similarity
            "min_brand_distance": self._min_brand_distance(sld),
            "min_brand_absdist": self._min_brand_absdist(sld),
            # Validators
            "is_valid_url": int(bool(validators.url(u))),
            "has_signin_token": int(bool(re.search(r'\b(sign|signin|login|auth)\b', u, flags=re.I))),
            "has_openid_token": int(bool(re.search(r'openid|oauth|token|saml', u, flags=re.I))),
            "percent_encoded": int('%' in u),
            # Heuristics
            "hyphen_count": host.count('-'),
            "suspicious_at": suspicious_at,
            # Trusted
            "is_trusted": int(any(host == td or host.endswith("."+td) for td in TRUSTED_DOMAINS)) if host else 0,
        }
        return feats

    def _min_brand_distance(self, sld):
        if not sld: return 1.0
        sld = unicodedata.normalize("NFKC", sld)
        return min(normalized_distance(sld, b) for b in BRANDS)

    def _min_brand_absdist(self, sld):
        if not sld: return max(len(b) for b in BRANDS)
        sld = unicodedata.normalize("NFKC", sld)
        return min(levenshtein(sld, b) for b in BRANDS)
