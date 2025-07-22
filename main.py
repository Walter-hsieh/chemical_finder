# Combined main.py for Streamlit web application

# --- config.py content ---
# Configuration file for the Chemical Research Agent

# API Keys (replace with your actual keys if needed)
# For Semantic Scholar, an API key is not strictly necessary for basic searches,
# but can provide higher rate limits.
SEMANTIC_SCHOLAR_API_KEY = None # "YOUR_SEMANTIC_SCHOLAR_API_KEY"

# Database Configuration
# Path to the SQLite database file
DATABASE_PATH = "data/chemicals.db"

# Search Limits
# Maximum number of papers to fetch from each source
MAX_PAPERS_PER_SOURCE = 10

# Caching
# Time-to-live (TTL) for cached data in seconds (e.g., 3600 seconds = 1 hour)
CACHE_TTL_SECONDS = 3600

# Other settings
# Enable/disable SSL verification for requests (set to False only for debugging/development)
VERIFY_SSL_REQUESTS = True

# Logging level (e.g., logging.INFO, logging.DEBUG)
LOGGING_LEVEL = "INFO"

# You can add more configuration variables as needed
# For example, proxy settings, user-agent strings, etc.


# --- database.py content ---
import sqlite3
import os
from datetime import datetime, timezone

def initialize_db():
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chemicals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_name TEXT,
                matched_name TEXT,
                cid TEXT,
                image_url TEXT,
                searched_at TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_searched_at ON chemicals(searched_at DESC)
        """)
        conn.commit()

def save_chemical(input_name, matched_name, cid, image_url):
    initialize_db()
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO chemicals (input_name, matched_name, cid, image_url, searched_at)
                VALUES (?, ?, ?, ?, ?)
            """, (input_name, matched_name, cid, image_url, datetime.now(timezone.utc)))
            conn.commit()
    except Exception as e:
        print(f"[ERROR] Failed to save chemical: {e}")

def load_history(limit=10):
    if not os.path.exists(DATABASE_PATH):
        return []
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT input_name, cid, searched_at FROM chemicals
                ORDER BY searched_at DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
        return rows
    except Exception as e:
        print(f"[ERROR] Failed to load history: {e}")
        return []

def clear_history():
    if not os.path.exists(DATABASE_PATH):
        return
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chemicals")
            conn.commit()
    except Exception as e:
        print(f"[ERROR] Failed to clear history: {e}")


# --- chemical_lookup.py content ---
import requests
import urllib3
import logging
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable SSL warnings in dev (not recommended in prod)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Use a session with retry logic
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

def fetch_pubchem_image(name_or_cas):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    try:
        cid_resp = session.get(f"{base_url}/compound/name/{name_or_cas}/cids/JSON", timeout=10)
        cid_resp.raise_for_status()
    except requests.exceptions.SSLError:
        logger.warning("SSL verification failed. Retrying with verify=False...")
        try:
            cid_resp = requests.get(f"{base_url}/compound/name/{name_or_cas}/cids/JSON", verify=False, timeout=10)
            cid_resp.raise_for_status()
        except Exception as e:
            logger.error(f"PubChem CID fetch failed after retry: {e}")
            return None, None, None, None
    except requests.exceptions.HTTPError as http_err:
        logger.warning(f"PubChem returned HTTP error: {http_err}")
        return None, None, None, None
    except Exception as e:
        logger.error(f"PubChem CID fetch failed: {e}")
        return None, None, None, None

    cids = cid_resp.json().get("IdentifierList", {}).get("CID", [])
    if not cids:
        logger.info("PubChem: No CID found.")
        return None, None, None, None
    cid = cids[0]

    try:
        name_resp = session.get(f"{base_url}/compound/cid/{cid}/property/IUPACName/JSON", timeout=10)
        name_resp.raise_for_status()
        props = name_resp.json().get("PropertyTable", {}).get("Properties", [])
        matched_name = props[0].get("IUPACName", name_or_cas) if props else name_or_cas
    except Exception:
        matched_name = name_or_cas

    encoded_name = quote(matched_name)
    image_url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_name}/image"

    return cid, image_url, "PubChem (Cactus)", matched_name

def fetch_cactus_image(name_or_cas):
    encoded_name = quote(name_or_cas)
    image_url = f"https://cactus.nci.nih.gov/chemical/structure/{encoded_name}/image"
    try:
        resp = session.head(image_url, timeout=5)
        if resp.status_code == 200:
            return None, image_url, "Cactus", name_or_cas
        else:
            logger.info(f"Cactus returned status {resp.status_code} for {name_or_cas}")
            return None, None, None, None
    except Exception as e:
        logger.error(f"Cactus request failed: {e}")
        return None, None, None, None

def fetch_wikidata(name_or_cas):
    search_url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": name_or_cas,
        "language": "en",
        "format": "json",
        "limit": 1,
        "type": "item"
    }
    try:
        resp = session.get(search_url, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("search", [])
        if results:
            entity = results[0]
            matched_name = entity.get("label", name_or_cas)
            wikidata_url = f"https://www.wikidata.org/wiki/{entity.get('id')}"
            return None, wikidata_url, "Wikidata", matched_name
        else:
            logger.info("Wikidata: No search results.")
            return None, None, None, None
    except Exception as e:
        logger.error(f"Wikidata fetch failed: {e}")
        return None, None, None, None

def fetch_chemical_info(name_or_cas):
    cid, image_url, source, matched_name = fetch_pubchem_image(name_or_cas)
    if cid or image_url:
        return cid, image_url, source, matched_name

    cid, image_url, source, matched_name = fetch_cactus_image(name_or_cas)
    if image_url:
        return cid, image_url, source, matched_name

    cid, image_url, source, matched_name = fetch_wikidata(name_or_cas)
    if image_url:
        return cid, image_url, source, matched_name

    return None, None, None, None


# --- paper_search.py content ---
import urllib.parse
import xml.etree.ElementTree as ET
import re
from html import unescape
from requests.exceptions import SSLError, RequestException

# logging is already configured above

# VERIFY_SSL is replaced by VERIFY_SSL_REQUESTS from config.py
MAX_RETRIES = 3 # Define MAX_RETRIES for paper_search context

def safe_get(url, timeout=10, verify=VERIFY_SSL_REQUESTS):
    try:
        return requests.get(url, timeout=timeout, verify=verify)
    except SSLError:
        if verify:
            logger.warning(f"SSL failed for {url}. Retrying with verify=False...")
            return safe_get(url, timeout=timeout, verify=False)
        raise

def clean_abstract(raw_abstract):
    if not raw_abstract:
        return "No abstract available."
    clean = re.sub('<[^<]+?>', '', raw_abstract)
    return unescape(clean).strip()

def parse_semantic_response(response):
    data = response.json().get("data", [])
    papers = []
    for paper in data:
        authors = ", ".join([a.get("name", "N/A") for a in paper.get("authors", [])])
        pdf_url = paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None
        papers.append({
            "title": paper.get("title", ""),
            "authors": authors,
            "year": paper.get("year", "N/A"),
            "url": paper.get("url", ""),
            "abstract": paper.get("abstract", "No abstract available."),
            "pdf_url": pdf_url
        })
    return papers

def search_semantic_scholar(query, limit=5):
    if not query.strip():
        return []
    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?"
        f"query={encoded_query}&limit={limit}&fields=title,authors,year,url,abstract,externalIds,openAccessPdf"
    )
    logger.debug(f"Semantic Scholar: {url}")
    for _ in range(MAX_RETRIES): # Using MAX_RETRIES from paper_search.py
        try:
            response = safe_get(url)
            if response.status_code == 200:
                return parse_semantic_response(response)[:limit]
        except Exception as e:
            logger.error(f"Semantic Scholar error: {e}")
    return []

def search_crossref(query, limit=5):
    if not query.strip():
        return []
    encoded_query = urllib.parse.quote(query)
    url = f"https://api.crossref.org/works?query={encoded_query}&rows={limit}"
    logger.debug(f"CrossRef: {url}")
    try:
        response = safe_get(url)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"CrossRef error: {e}")
        return []

    items = response.json().get("message", {}).get("items", [])
    results = []
    for item in items:
        authors = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])]
        authors_str = ", ".join(authors) if authors else "N/A"
        pdf_url = next((link.get("URL") for link in item.get("link", []) if link.get("content-type") == "application/pdf"), None)
        results.append({
            "title": item.get("title", [""])[0],
            "authors": authors_str,
            "year": item.get("issued", {}).get("date-parts", [[None]])[0][0],
            "url": item.get("URL", ""),
            "abstract": clean_abstract(item.get("abstract")),
            "pdf_url": pdf_url
        })
    return results[:limit]

def search_arxiv(query, limit=5):
    if not query.strip():
        return []
    base_url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&start=0&max_results={limit}"
    logger.debug(f"arXiv: {base_url}")
    try:
        response = safe_get(base_url)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"arXiv error: {e}")
        return []

    entries = []
    try:
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns).text.strip()
            abstract = entry.find("atom:summary", ns).text.strip()
            url = entry.find("atom:id", ns).text
            year = entry.find("atom:published", ns).text[:4]
            arxiv_id = url.split('/abs/')[-1]
            pdf_url = f"http://arxiv.org/pdf/{arxiv_id}.pdf"
            authors = [author.text for author in entry.findall("atom:author/atom:name", ns)]
            authors_str = ", ".join(authors) or "N/A"
            entries.append({
                "title": title,
                "authors": authors_str,
                "year": year,
                "url": url,
                "abstract": abstract,
                "pdf_url": pdf_url
            })
    except Exception as e:
        logger.error(f"Failed to parse arXiv response: {e}")
        return []

    return entries[:limit]

def search_papers(query, limit=10):
    all_papers = []
    for source, func in [
        ("Semantic Scholar", search_semantic_scholar),
        ("CrossRef", search_crossref),
        ("arXiv", search_arxiv)
    ]:
        papers = func(query, limit=limit - len(all_papers))
        if papers:
            all_papers.extend(papers)
            if len(all_papers) >= limit:
                return all_papers[:limit], "Multiple Sources"
    return all_papers, "Multiple Sources" if all_papers else "None"


# --- utils.py content ---
# You can add any helper functions here


# --- app.py content (main Streamlit application logic) ---
import streamlit as st
import pandas as pd # Already imported
# from chemical_lookup import fetch_chemical_info # Removed, now in same file
# from paper_search import search_papers # Removed, now in same file
# from database import save_chemical, load_history, clear_history # Removed, now in same file
# from datetime import datetime # Already imported
# from urllib.parse import quote # Already imported
# import requests # Already imported

st.set_page_config(page_title="Chemical Research Agent", layout="centered")
st.title("🧪 Chemical Research Agent")

@st.cache_data(ttl=CACHE_TTL_SECONDS) # Using CACHE_TTL_SECONDS from config
def get_papers(query):
    return search_papers(query)

chemical_input = st.text_input("Enter chemical name or CAS number:")

if st.button("🔍 Search"):
    if not chemical_input.strip():
        st.warning("Please enter a valid chemical name.")
    else:
        with st.spinner("Fetching chemical info and related papers..."):
            cid, image_url, source, matched_name = fetch_chemical_info(chemical_input)
            search_term = matched_name or chemical_input

            if cid or image_url:
                if image_url:
                    st.image(image_url, caption=f"{search_term} (Source: {source})", use_container_width=False)
                if cid:
                    st.success(f"CID: {cid}")

                if matched_name and matched_name.lower() != chemical_input.lower():
                    st.info(f"Matched to: `{matched_name}`")

                save_chemical(chemical_input, matched_name or "", cid or "N/A", image_url or "")

                st.subheader("📚 Related Papers")
                st.caption(f"Searching papers for: `{search_term}`")

                try:
                    papers, source_used = get_papers(search_term)
                    source_links = {
                        "Semantic Scholar": "https://www.semanticscholar.org/",
                        "CrossRef": "https://search.crossref.org/",
                        "arXiv": "https://arxiv.org/"
                    }
                    st.markdown(f"✅ Showing results from [{source_used}]({source_links.get(source_used, '#')})")

                except Exception as e:
                    st.error("❌ Paper search failed. Please check your internet connection or try again later.")
                    st.exception(e)
                    papers = []

                if papers:
                    paper_list = []
                    for p in papers:
                        p['source'] = source_used  # add source to paper dict
                        st.markdown(f"**{p['title']}** ({p['year']})")
                        st.markdown(f"👨‍🔬 *{p['authors']}*")
                        st.markdown(f"[🔗 Read more]({p['url']})")
                        st.markdown(p.get('abstract', '_No abstract available._'))

                        pdf_url = p.get("pdf_url")
                        if pdf_url:
                            st.markdown(f"[📄 Download PDF]({pdf_url})")

                        st.markdown("---")
                        paper_list.append(p)

                    df = pd.DataFrame(paper_list)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Papers as CSV",
                        data=csv,
                        file_name=f"papers_{search_term.replace(' ', '_')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No related papers found. Try different keywords, synonyms, or identifiers.")
            else:
                st.warning("🔍 Compound not found in structured chemical databases.")
                query_encoded = quote(chemical_input)
                pubchem_url = f"https://pubchem.ncbi.nlm.nih.gov/#query={query_encoded}"
                wikidata_url = f"https://www.wikidata.org/w/index.php?search={query_encoded}"

                st.info(
                    f"""⚠️ This compound may still exist in scientific literature, but no structure or CID was found.

💡 Try entering a more precise identifier such as:
- CAS number (e.g. `50-00-0`)
- IUPAC name (e.g. `methanal`)
- SMILES notation (e.g. `C=O`)

You can also try searching manually:
- 🔬 [Search on PubChem]({pubchem_url})
- 🧠 [Search on Wikidata]({wikidata_url})"""
                )

st.divider()
st.markdown("### 🔁 Recent Search History")

if st.button("🗑️ Clear History"):
    clear_history()
    st.success("Search history cleared!")

history = load_history(limit=10)
if history:
    for name, cid, time in history:
        time_str = datetime.fromisoformat(time).strftime("%Y-%m-%d %H:%M")
        st.markdown(f"🧪 **{name}** — CID: `{cid}`  _(at {time_str})_")
else:
    st.info("No search history yet.")