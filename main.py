# chemical_finder_app.py (with sorting fix)
import streamlit as st
import pandas as pd
import sqlite3
import os
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import quote, urlparse
from html import unescape
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 1. Centralized Configuration ---
class Config:
    DATABASE_PATH = "data/chemicals.db"
    MAX_PAPERS_PER_SOURCE = 5
    CACHE_TTL_SECONDS = 3600
    VERIFY_SSL_REQUESTS = True
    REQUEST_TIMEOUT_SECONDS = 10

# --- 2. Centralized Session Management ---
def create_requests_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.verify = Config.VERIFY_SSL_REQUESTS
    session.headers.update({'User-Agent': 'ChemicalResearchAgent/1.0'})
    return session

api_session = create_requests_session()

# --- 3. Database Module ---
class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialize_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _initialize_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chemicals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_name TEXT, matched_name TEXT, cid TEXT, image_url TEXT, searched_at TEXT
                )
            """)
            conn.commit()

    def save_chemical(self, input_name, matched_name, cid, image_url):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            timestamp_str = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                "INSERT INTO chemicals (input_name, matched_name, cid, image_url, searched_at) VALUES (?, ?, ?, ?, ?)",
                (input_name, matched_name, cid, image_url, timestamp_str)
            )
            conn.commit()

    def load_history(self, limit=10):
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT input_name, cid, searched_at FROM chemicals ORDER BY searched_at DESC LIMIT ?", (limit,))
                return cursor.fetchall()
            except sqlite3.OperationalError:
                return []

    def clear_history(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chemicals")
            conn.commit()

# --- 4. API Client Modules ---
class ChemicalFinder:
    def fetch_pubchem(self, name):
        base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
        try:
            cid_resp = api_session.get(f"{base_url}/compound/name/{name}/cids/JSON", timeout=Config.REQUEST_TIMEOUT_SECONDS)
            if cid_resp.status_code != 200: return None
            cid = cid_resp.json().get("IdentifierList", {}).get("CID", [])[0]
            
            name_resp = api_session.get(f"{base_url}/compound/cid/{cid}/property/IUPACName/JSON", timeout=Config.REQUEST_TIMEOUT_SECONDS)
            props = name_resp.json().get("PropertyTable", {}).get("Properties", [])
            matched_name = props[0].get("IUPACName", name) if props else name
            
            image_url = f"https://cactus.nci.nih.gov/chemical/structure/{quote(matched_name)}/image"
            return {"cid": str(cid), "image_url": image_url, "source": "PubChem", "matched_name": matched_name}
        except (requests.RequestException, IndexError, KeyError):
            return None

    def fetch_cactus(self, name):
        image_url = f"https://cactus.nci.nih.gov/chemical/structure/{quote(name)}/image"
        try:
            resp = api_session.head(image_url, timeout=Config.REQUEST_TIMEOUT_SECONDS)
            if resp.status_code == 200:
                return {"cid": None, "image_url": image_url, "source": "Cactus", "matched_name": name}
        except requests.RequestException:
            return None
        return None

    def find_best_info(self, name):
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_source = {executor.submit(self.fetch_pubchem, name): "PubChem", executor.submit(self.fetch_cactus, name): "Cactus"}
            results = {}
            for future in as_completed(future_to_source):
                source = future_to_source[future]
                result = future.result()
                if result: results[source] = result
            return results.get("PubChem") or results.get("Cactus")

class PaperFinder:
    def _clean_abstract(self, raw_abstract):
        if not raw_abstract: return "No abstract available."
        clean = re.sub('<[^<]+?>', '', raw_abstract)
        return unescape(clean).strip()

    def search_semantic_scholar(self, query, limit):
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={quote(query)}&limit={limit}&fields=title,authors,year,url,abstract,openAccessPdf"
        try:
            response = api_session.get(url, timeout=Config.REQUEST_TIMEOUT_SECONDS)
            if response.status_code != 200: return []
            data = response.json().get("data", [])
            papers = []
            for paper in data:
                pdf_url = paper.get("openAccessPdf", {}).get("url") if paper.get("openAccessPdf") else None
                papers.append({
                    "title": paper.get("title", ""), "authors": ", ".join([a.get("name", "N/A") for a in paper.get("authors", [])]),
                    "year": paper.get("year", None), "url": paper.get("url", ""), "abstract": paper.get("abstract", "No abstract."), "pdf_url": pdf_url
                })
            return papers
        except requests.RequestException: return []

    def search_crossref(self, query, limit):
        url = f"https://api.crossref.org/works?query={quote(query)}&rows={limit}"
        try:
            response = api_session.get(url, timeout=Config.REQUEST_TIMEOUT_SECONDS)
            if response.status_code != 200: return []
            items = response.json().get("message", {}).get("items", [])
            results = []
            for item in items:
                authors_str = ", ".join([f"{a.get('given', '')} {a.get('family', '')}".strip() for a in item.get("author", [])])
                pdf_url = next((link.get("URL") for link in item.get("link", []) if link.get("content-type") == "application/pdf"), None)
                results.append({
                    "title": item.get("title", [""])[0], "authors": authors_str, "year": item.get("issued", {}).get("date-parts", [[None]])[0][0],
                    "url": item.get("URL", ""), "abstract": self._clean_abstract(item.get("abstract")), "pdf_url": pdf_url
                })
            return results
        except requests.RequestException: return []

    def search_arxiv(self, query, limit):
        url = f"http://export.arxiv.org/api/query?search_query=all:{quote(query)}&start=0&max_results={limit}"
        try:
            response = api_session.get(url, timeout=Config.REQUEST_TIMEOUT_SECONDS)
            if response.status_code != 200: return []
            root = ET.fromstring(response.text)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = []
            for entry in root.findall("atom:entry", ns):
                arxiv_id = urlparse(entry.find("atom:id", ns).text).path.strip("/abs/")
                entries.append({
                    "title": entry.find("atom:title", ns).text.strip(), "authors": ", ".join([a.text for a in entry.findall("atom:author/atom:name", ns)]),
                    "year": int(entry.find("atom:published", ns).text[:4]), "url": entry.find("atom:id", ns).text,
                    "abstract": entry.find("atom:summary", ns).text.strip(), "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                })
            return entries
        except (requests.RequestException, ET.ParseError): return []

    def search_all(self, query, limit=10):
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_searches = {
                executor.submit(self.search_semantic_scholar, query, limit): "Semantic Scholar",
                executor.submit(self.search_crossref, query, limit): "CrossRef",
                executor.submit(self.search_arxiv, query, limit): "arXiv"
            }
            all_papers = []
            for future in as_completed(future_searches):
                papers = future.result()
                if papers: all_papers.extend(papers)
            
            unique_papers = {p['title'].lower().strip(): p for p in all_papers}.values()
            # --- FIX: Use 'or 0' to handle None values for 'year' during sorting ---
            return sorted(list(unique_papers), key=lambda x: x.get('year') or 0, reverse=True)[:limit]

# --- 5. Main Streamlit Application ---
st.set_page_config(page_title="Chemical Research Agent", layout="wide")
st.title("🧪 Chemical Research Agent")

# Initialize managers
db = DatabaseManager(Config.DATABASE_PATH)
chemical_finder = ChemicalFinder()
paper_finder = PaperFinder()

@st.cache_data(ttl=Config.CACHE_TTL_SECONDS)
def get_chemical_info(name):
    return chemical_finder.find_best_info(name)

@st.cache_data(ttl=Config.CACHE_TTL_SECONDS)
def get_papers(query):
    return paper_finder.search_all(query)

# --- Streamlit UI ---
chemical_input = st.text_input("Enter chemical name, CAS number, or SMILES:", key="chem_input")

if st.button("🔍 Search", key="search_button"):
    if chemical_input:
        with st.spinner("Concurrently searching databases and literature..."):
            chem_info = get_chemical_info(chemical_input)
            
            if chem_info:
                search_term = chem_info["matched_name"]
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(chem_info["image_url"], caption=f"Source: {chem_info['source']}", use_container_width=True)
                with col2:
                    st.subheader(search_term)
                    if chem_info["cid"]: st.success(f"PubChem CID: {chem_info['cid']}")
                    if search_term.lower() != chemical_input.lower(): st.info(f"Input was matched to: `{search_term}`")
                
                db.save_chemical(chemical_input, search_term, chem_info["cid"] or "N/A", chem_info["image_url"])
            else:
                st.warning("Compound not found in databases. Searching papers with the provided input.")
                search_term = chemical_input

            st.subheader(f"📚 Related Papers for `{search_term}`")
            papers = get_papers(search_term)
            
            if papers:
                for p in papers:
                    st.markdown(f"**{p['title']}** ({p.get('year', 'N/A')})")
                    st.markdown(f"👨‍🔬 *{p.get('authors', 'N/A')}*")
                    if p.get('url'): st.markdown(f"[🔗 Read more]({p['url']})")
                    if p.get('pdf_url'): st.markdown(f"[📄 Download PDF]({p['pdf_url']})")
                    with st.expander("View Abstract"):
                        st.markdown(p.get('abstract', '_No abstract available._'))
                    st.markdown("---")
                
                df = pd.DataFrame(papers)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(label="⬇️ Download Papers as CSV", data=csv, file_name=f"papers_{search_term.replace(' ', '_')}.csv", mime="text/csv")
            else:
                st.warning("No related papers found.")

# History display
st.divider()
st.subheader("🔁 Recent Search History")
if st.button("🗑️ Clear History"):
    db.clear_history()
    st.success("History cleared!")
    st.rerun()

history = db.load_history()
if history:
    for name, cid, time_str in history:
        try:
            dt_obj = datetime.fromisoformat(time_str)
            formatted_time = dt_obj.strftime("%Y-%m-%d %H:%M UTC")
            st.write(f"- **{name}** (CID: {cid}) - Searched on {formatted_time}")
        except (ValueError, TypeError):
            st.write(f"- **{name}** (CID: {cid}) - Invalid timestamp in DB")
else:
    st.info("No searches yet.")