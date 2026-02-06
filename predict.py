# predict.py
import os
import json
import re
from urllib.parse import urlparse

import pandas as pd
from scipy import sparse
from joblib import load

from feature_extraction import FeatureExtraction, TRUSTED_DOMAINS, BRANDS
from reputation_utils import (
    expand_redirects,
    get_certificate_info,
    domain_age_days,
    google_safe_browsing_lookup,
    virustotal_lookup,
)

# ----------------- Load artifacts (must exist after training) ----------------- #
tfidf = load('tfidf_vectorizer.joblib')
scaler = load('scaler.joblib')
model = load('phishing_model.joblib')

with open('engineered_feature_names.json', 'r') as f:
    eng_feature_names = list(json.load(f).keys())

# ----------------- API keys (env first, fallback if present) ----------------- #
# Recommended: set environment variables instead of hardcoding.
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', "AIzaSyB_cwGKRaofJXoBV3CD9BEiiIUB8FeDZCg")
VT_API_KEY = os.environ.get('VT_API_KEY', None)  # set to your VirusTotal key if available

# ----------------- Tuning params ----------------- #
TYPOSQUAT_NORM_DIST_THRESHOLD = 0.30   # normalized Levenshtein distance threshold (lower -> stricter)
NEW_DOMAIN_AGE_DAYS = 30               # domains younger than this flagged in network checks
SUSPICIOUS_HOST_PROVIDERS = {"trycloudflare.com", "ngrok.io", "serveo.net", "loca.lt", "repl.co"}

# ----------------- Helpers ----------------- #
def engineered_df_from_urls(urls):
    feats = [FeatureExtraction(u).to_dict() for u in urls]
    df = pd.DataFrame(feats).fillna(0)
    # ensure consistent column order / presence
    for c in eng_feature_names:
        if c not in df.columns:
            df[c] = 0
    return df[eng_feature_names]

def heuristic_flags_from_host(hostname):
    parts = hostname.split('.') if hostname else []
    return {
        'long_subdomain': int(len(parts) > 3),
        'many_hyphens': int(hostname.count('-') > 3),
        'suspicious_provider': int(any(hostname == p or hostname.endswith("." + p) for p in SUSPICIOUS_HOST_PROVIDERS)),
        'has_digit_tail': int(any(ch.isdigit() for ch in (hostname[-6:] if hostname else ""))),
        'punycode': int('xn--' in hostname),
    }

# ----------------- Main prediction function ----------------- #
def predict_urls(urls, do_network_checks=False, safe_browsing_key=None, vt_key=None):
    """
    Predict list of raw URLs.
    - urls: list of raw URL strings (may omit scheme)
    - do_network_checks: if True, run redirects/cert/whois checks (slower)
    - safe_browsing_key / vt_key: optional API keys (if not provided, env keys are used)
    Returns: list of tuples (original_url, pred_int, prob_float)
    """
    # Resolve API key inputs (priority: function arg -> env var -> default)
    sb_key = safe_browsing_key if safe_browsing_key is not None else GOOGLE_API_KEY
    vt_key_effective = vt_key if vt_key is not None else VT_API_KEY

    # Normalize input (add scheme if missing) for vectorizer & feature extraction
    urls_with_scheme = [u if u.startswith(('http://','https://')) else 'http://' + u for u in urls]

    # Compute TF-IDF and engineered features once
    X_tfidf = tfidf.transform(urls_with_scheme)
    X_eng = engineered_df_from_urls(urls_with_scheme)
    X_eng_scaled = scaler.transform(X_eng)
    X_combined = sparse.hstack([X_tfidf, sparse.csr_matrix(X_eng_scaled)], format='csr')

    # ML predictions
    proba = model.predict_proba(X_combined)[:, 1]
    pred = (proba >= 0.5).astype(int)

    results = []
    for i, raw in enumerate(urls_with_scheme):
        orig = urls[i]
        parsed = urlparse(raw)
        host = (parsed.hostname or "").lower()

        # --- 1) Trusted domain exact -> force LEGIT (whitelist)
        if host and any(host == td or host.endswith("." + td) for td in TRUSTED_DOMAINS):
            results.append((orig, 0, 0.0))
            continue

        # --- 2) Fast typosquat override (using engineered feature)
        min_brand_dist = X_eng.iloc[i].get('min_brand_distance', 1.0)
        if min_brand_dist < TYPOSQUAT_NORM_DIST_THRESHOLD:
            # Strong override to phishing
            results.append((orig, 1, max(proba[i], 0.98)))
            continue

        # --- 3) Simple heuristics (long subdomain, hyphens, suspicious provider, punycode)
        heur = heuristic_flags_from_host(host)
        if heur['long_subdomain'] or heur['many_hyphens'] or heur['suspicious_provider'] or heur['punycode']:
            results.append((orig, 1, max(proba[i], 0.90)))
            continue

        # --- 4) Optional network / reputation checks (slower)
        if do_network_checks:
            # 4a. Redirect expansion
            try:
                chain, status = expand_redirects(raw)
            except Exception:
                chain = []
            final = chain[-1] if chain else raw
            final_host = urlparse(final).hostname or ""

            # If final redirect is an IP -> suspicious
            if re.match(r'^\d+\.\d+\.\d+\.\d+$', final_host):
                results.append((orig, 1, max(proba[i], 0.95)))
                continue

            # 4b. WHOIS / domain age
            try:
                age = domain_age_days(final_host)
                if age is not None and age < NEW_DOMAIN_AGE_DAYS:
                    results.append((orig, 1, max(proba[i], 0.88)))
                    continue
            except Exception:
                pass

            # 4c. Certificate SAN check
            try:
                cert = get_certificate_info(final_host)
                if cert:
                    san = cert.get('subjectAltName', []) if isinstance(cert, dict) else []
                    san_hosts = [t[1] for t in san] if san else []
                    if final_host and san_hosts and not any(final_host.endswith(x) for x in san_hosts):
                        results.append((orig, 1, max(proba[i], 0.90)))
                        continue
            except Exception:
                pass

            # 4d. Google Safe Browsing (if key present)
            try:
                gs = google_safe_browsing_lookup(raw, sb_key)
                if gs is True:
                    results.append((orig, 1, 0.99))
                    continue
            except Exception:
                pass

            # 4e. VirusTotal (if key present)
            try:
                vt = virustotal_lookup(raw, vt_key_effective)
                if vt is True:
                    results.append((orig, 1, 0.98))
                    continue
            except Exception:
                pass

        # --- 5) Fallback to ML probability
        results.append((orig, int(pred[i]), float(proba[i])))

    return results

# ----------------- Quick test (only when running directly) ----------------- #
if __name__ == "__main__":
    examples = [
        "https://www.amazon.in/ap/signin?openid.pape.max_auth_age=0",
        "https://www.amaaazon.in/ap/signin?openid.pape.max_auth_age=0",
        "https://hello-insulation-sweet-gary.trycloudflare.com",
        "https://www.google.com"
    ]
    out = predict_urls(examples, do_network_checks=False)
    for url, p, prob in out:
        print(f"{url} -> {'PHISHING' if p==1 else 'LEGIT'} ({prob:.3f})")
