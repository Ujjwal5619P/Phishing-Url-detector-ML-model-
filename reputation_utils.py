# reputation_utils.py
import requests
import ssl, socket
from urllib.parse import urlparse
import time
from functools import lru_cache
import base64

# whois is optional (pip install python-whois)
try:
    import whois
except Exception:
    whois = None

# Basic redirect expand (fast, with timeout)
def expand_redirects(url, timeout=5):
    try:
        r = requests.get(url, allow_redirects=True, timeout=timeout)
        chain = [resp.url for resp in r.history] + [r.url]
        return chain, r.status_code
    except Exception:
        return [], None

# TLS certificate: returns subject CN and SANs or None
def get_certificate_info(hostname, port=443, timeout=5):
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                return cert
    except Exception:
        return None

# WHOIS age in days (may fail / rate limit)
@lru_cache(maxsize=10000)
def domain_age_days(domain):
    if whois is None:
        return None
    try:
        w = whois.whois(domain)
        if hasattr(w, "creation_date"):
            cd = w.creation_date
            if isinstance(cd, list):
                cd = cd[0]
            if not cd:
                return None
            delta = (time.time() - cd.timestamp())/86400.0
            return max(0, int(delta))
    except Exception:
        return None

# ----------------- Google Safe Browsing -----------------
def google_safe_browsing_lookup(url, api_key):
    """
    Returns:
      True  -> Google flags url as unsafe
      False -> Google does not flag url
      None  -> api_key not provided or API error
    """
    if not api_key:
        return None
    endpoint = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={api_key}"
    payload = {
        "client": {"clientId": "phish-detector", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "POTENTIALLY_HARMFUL_APPLICATION", "UNWANTED_SOFTWARE"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}]
        }
    }
    try:
        r = requests.post(endpoint, json=payload, timeout=6)
        if r.status_code == 200:
            data = r.json()
            # If 'matches' present (non-empty), it's flagged
            if data:
                return True
            return False
        # non-200 -> treat as None (unknown)
        return None
    except Exception:
        return None

# ----------------- VirusTotal lookup (v3) -----------------
def virustotal_lookup(url, api_key):
    """
    Returns:
      True  -> flagged as malicious by VT
      False -> not flagged (or returned zero malicious)
      None  -> no api_key / error
    Note: VirusTotal v3 expects the URL to be encoded for the /urls/{id} endpoint.
    """
    if not api_key:
        return None
    try:
        # encode URL to url_id per VT spec
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        headers = {"x-apikey": api_key}
        resp = requests.get(f"https://www.virustotal.com/api/v3/urls/{url_id}", headers=headers, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
            stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
            malicious_count = stats.get("malicious", 0)
            return malicious_count > 0
        # If 404 or other -> return False (not found) or None on error codes
        if resp.status_code == 404:
            return False
        return None
    except Exception:
        return None
