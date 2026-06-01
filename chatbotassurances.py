from __future__ import annotations

import html
import json
import re
import tempfile
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "code_cima_articles.json"
CIMA_PDF_URL = "https://cima-afrique.org/wp-content/uploads/2023/06/CODE-CIMA-2019.pdf"
MIN_CONFIDENCE = 0.10
MIN_TERM_COVERAGE = 0.34
DEFAULT_SOURCE_COUNT = 3

FRENCH_STOPWORDS = {
    "a",
    "afin",
    "ainsi",
    "alors",
    "au",
    "aucun",
    "aussi",
    "autre",
    "aux",
    "avec",
    "avoir",
    "ce",
    "ces",
    "cet",
    "cette",
    "combien",
    "comment",
    "comme",
    "d",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "elles",
    "en",
    "est",
    "et",
    "etre",
    "fait",
    "il",
    "ils",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "lors",
    "mais",
    "ne",
    "ou",
    "par",
    "parler",
    "pas",
    "peux",
    "peut",
    "pese",
    "pesent",
    "pour",
    "qu",
    "quand",
    "que",
    "quel",
    "quelle",
    "quelles",
    "quels",
    "qui",
    "s",
    "sans",
    "se",
    "ses",
    "son",
    "sont",
    "sur",
    "tu",
    "un",
    "une",
    "y",
}

BROAD_TERMS = {
    "assurance",
    "assurances",
    "assure",
    "assuree",
    "assurees",
    "assureur",
    "assureurs",
    "article",
    "articles",
    "cima",
    "code",
    "contrat",
    "contrats",
    "obligation",
    "obligations",
}

ARTICLE_QUERY_RE = re.compile(
    r"\barticles?\s+((?:\d+(?:-\d+)*(?:\s*bis)?|1er|premier|et|,|\s)+)",
    re.IGNORECASE,
)
ARTICLE_NUMBER_RE = re.compile(
    r"\b(?:\d+(?:-\d+)*(?:\s*bis)?|1er|premier)\b",
    re.IGNORECASE,
)


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return value.lower()


def canonical_article_number(value: str) -> str:
    value = normalize_text(value).strip()
    if value in {"premier", "1er"}:
        return "1"
    value = re.sub(r"\s+", " ", value)
    value = value.replace(" bis", " bis")
    return value


def extract_requested_articles(query: str) -> list[str]:
    requested: list[str] = []
    for block in ARTICLE_QUERY_RE.findall(query):
        for match in ARTICLE_NUMBER_RE.findall(block):
            number = canonical_article_number(match)
            if number not in requested:
                requested.append(number)
    return requested


def tokenize(value: str) -> set[str]:
    normalized = normalize_text(value)
    tokens = re.findall(r"[a-z0-9]{3,}", normalized)
    return {token for token in tokens if token not in FRENCH_STOPWORDS}


def get_specific_terms(query: str) -> set[str]:
    return {term for term in tokenize(query) if term not in BROAD_TERMS}


@st.cache_data(show_spinner=False)
def load_article_payload() -> dict[str, Any]:
    if DATA_PATH.exists():
        with DATA_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    from scripts.build_code_cima_index import build_payload

    cache_dir = Path(tempfile.gettempdir()) / "assistant_code_cima"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = cache_dir / "CODE-CIMA-2019.pdf"
    if not pdf_path.exists():
        urllib.request.urlretrieve(CIMA_PDF_URL, pdf_path)
    return build_payload(pdf_path, page_start=37, page_end=408)


@st.cache_resource(show_spinner=False)
def build_search_index(articles: tuple[tuple[str, str, str, str], ...]):
    titles = [
        f"Article {number}. {title}. {section}"
        for number, title, section, _body in articles
    ]
    full_texts = [
        f"Article {number}. {title}. {section}. {body}"
        for number, title, section, body in articles
    ]
    title_vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        ngram_range=(1, 3),
        stop_words=sorted(FRENCH_STOPWORDS),
        sublinear_tf=True,
    )
    title_char_vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(4, 6),
        sublinear_tf=True,
    )
    full_vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        ngram_range=(1, 2),
        stop_words=sorted(FRENCH_STOPWORDS),
        sublinear_tf=True,
    )
    return {
        "title_vectorizer": title_vectorizer,
        "title_matrix": title_vectorizer.fit_transform(titles),
        "title_char_vectorizer": title_char_vectorizer,
        "title_char_matrix": title_char_vectorizer.fit_transform(titles),
        "full_vectorizer": full_vectorizer,
        "full_matrix": full_vectorizer.fit_transform(full_texts),
        "article_terms": tuple(tokenize(text) for text in full_texts),
    }


def get_article_tuples(payload: dict[str, Any]) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (
            article["article"],
            article.get("title", ""),
            article.get("section", ""),
            article.get("body", ""),
        )
        for article in payload["articles"]
    )


def search_articles(
    query: str,
    payload: dict[str, Any],
    source_count: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    articles = payload["articles"]
    article_by_number = {
        canonical_article_number(article["article"]): article for article in articles
    }
    article_tuples = get_article_tuples(payload)
    index = build_search_index(article_tuples)

    warnings: list[str] = []
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for requested in extract_requested_articles(query):
        article = article_by_number.get(requested)
        if article:
            results.append(
                {
                    "article": article,
                    "score": 1.0,
                    "coverage": 1.0,
                    "reason": "Article demandé explicitement",
                }
            )
            seen.add(canonical_article_number(article["article"]))
        else:
            warnings.append(f"L'article {requested} n'a pas été trouvé dans l'index.")

    specific_terms = get_specific_terms(query)
    if results and not specific_terms:
        return results[:source_count], warnings

    title_scores = cosine_similarity(
        index["title_vectorizer"].transform([query]), index["title_matrix"]
    ).ravel()
    title_char_scores = cosine_similarity(
        index["title_char_vectorizer"].transform([query]), index["title_char_matrix"]
    ).ravel()
    full_scores = cosine_similarity(
        index["full_vectorizer"].transform([query]), index["full_matrix"]
    ).ravel()
    scores = (0.5 * title_scores) + (0.2 * title_char_scores) + (0.3 * full_scores)

    coverages = np.ones_like(scores)
    if specific_terms:
        adjusted_scores = scores.copy()
        for index_number, article_terms in enumerate(index["article_terms"]):
            coverage = len(specific_terms & article_terms) / len(specific_terms)
            coverages[index_number] = coverage
            if coverage == 0:
                adjusted_scores[index_number] *= 0.25
            elif coverage < 0.5:
                adjusted_scores[index_number] *= 0.65
            adjusted_scores[index_number] += 0.04 * coverage
        scores = adjusted_scores

    best_indexes = np.argsort(scores)[::-1][: max(source_count * 3, 8)]

    for index in best_indexes:
        article = articles[int(index)]
        number = canonical_article_number(article["article"])
        if number in seen:
            continue
        score = float(scores[int(index)])
        if score <= 0:
            continue
        results.append(
            {
                "article": article,
                "score": score,
                "coverage": float(coverages[int(index)]),
                "reason": "Correspondance sémantique",
            }
        )
        seen.add(number)
        if len(results) >= source_count:
            break

    return results[:source_count], warnings


def split_sentences(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    sentences = re.split(r"(?<=[.;:!?])\s+(?=[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜ0-9])", compact)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def select_excerpts(article: dict[str, Any], query: str, max_sentences: int = 3) -> list[str]:
    sentences = split_sentences(article.get("body", ""))
    if not sentences:
        return []

    terms = tokenize(query)
    scored: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(sentences):
        sentence_terms = tokenize(sentence)
        overlap = len(terms & sentence_terms)
        if overlap:
            scored.append((overlap, index, sentence))

    if scored:
        selected = sorted(scored, key=lambda item: (-item[0], item[1]))[:max_sentences]
        return [sentence for _, _, sentence in sorted(selected, key=lambda item: item[1])]

    return sentences[:max_sentences]


def format_pages(article: dict[str, Any]) -> str:
    start = article.get("page_start")
    end = article.get("page_end", start)
    if not start:
        return "page non renseignée"
    if start == end:
        return f"p. {start}"
    return f"p. {start}-{end}"


def format_reference(article: dict[str, Any]) -> str:
    title = article.get("title", "").strip()
    label = f"Article {article['article']}"
    if title:
        label += f" - {title}"
    return f"{label} ({format_pages(article)})"


def render_article(match: dict[str, Any], query: str, expanded: bool) -> None:
    article = match["article"]
    score = match["score"]
    reference = format_reference(article)
    confidence = "demande directe" if score == 1.0 else f"score {score:.2f}"
    coverage = match.get("coverage", 1.0)

    with st.expander(reference, expanded=expanded):
        section = article.get("section")
        if section:
            st.caption(section)
        st.markdown(f"**Type de correspondance :** {match['reason']} ({confidence})")
        if score != 1.0:
            st.caption(f"Couverture des termes importants : {coverage:.0%}")
        excerpts = select_excerpts(article, query)
        if excerpts:
            st.markdown("**Extraits utiles :**")
            for excerpt in excerpts:
                st.markdown(f"- {excerpt}")
        else:
            st.info("Aucun extrait exploitable n'a été trouvé pour cet article.")
        st.markdown("**Texte intégral de l'article indexé :**")
        st.write(article.get("body", ""))


def build_answer_text(query: str, matches: list[dict[str, Any]]) -> str:
    principal = matches[0]["article"]
    excerpts = select_excerpts(principal, query, max_sentences=2)
    reference = format_reference(principal)
    if not excerpts:
        return (
            f"Je trouve comme référence principale {reference}. "
            "Consultez l'extrait source ci-dessous avant toute décision."
        )
    excerpt_text = " ".join(excerpts)
    return (
        f"Réponse fondée sur {reference}. "
        f"{excerpt_text} "
        "La formulation ci-dessus reprend le contenu indexé du Code CIMA fourni."
    )


def render_response(message: dict[str, Any]) -> None:
    matches = message["matches"]
    warnings = message.get("warnings", [])
    status = message["status"]

    if warnings:
        for warning in warnings:
            st.warning(warning)

    if status == "low_confidence":
        st.warning(
            "Je ne trouve pas d'article suffisamment proche pour répondre de façon sûre. "
            "Reformulez avec des termes du Code CIMA ou indiquez un numéro d'article."
        )
        if matches:
            st.markdown("Articles proches à vérifier :")
            for match in matches:
                render_article(match, message["query"], expanded=False)
        return

    st.markdown(message["answer"])
    st.info(
        "Réponse documentaire basée sur l'édition 2019 fournie. "
        "Elle ne remplace pas l'analyse d'un professionnel du droit ou de l'assurance."
    )

    for index, match in enumerate(matches):
        render_article(match, message["query"], expanded=index == 0)


def make_response(query: str, payload: dict[str, Any], source_count: int) -> dict[str, Any]:
    matches, warnings = search_articles(query, payload, source_count)
    top_score = matches[0]["score"] if matches else 0.0
    top_coverage = matches[0].get("coverage", 0.0) if matches else 0.0
    direct_article = bool(matches and matches[0]["score"] == 1.0)
    status = (
        "ok"
        if direct_article
        or (matches and top_score >= MIN_CONFIDENCE and top_coverage >= MIN_TERM_COVERAGE)
        else "low_confidence"
    )
    answer = build_answer_text(query, matches) if status == "ok" else ""
    return {
        "query": query,
        "status": status,
        "answer": answer,
        "matches": matches,
        "warnings": warnings,
    }


def inject_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: #f6f8fb;
        }
        .block-container {
            max-width: 1120px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .cima-hero {
            background: #ffffff;
            border: 1px solid #d8e0ee;
            border-radius: 8px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(27, 39, 75, 0.06);
        }
        .cima-hero h1 {
            font-size: 2rem;
            line-height: 1.15;
            margin: 0 0 .45rem 0;
            color: #172033;
        }
        .cima-hero p {
            margin: 0;
            color: #475569;
            max-width: 760px;
        }
        .cima-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: .5rem;
            margin-top: .9rem;
        }
        .cima-pill {
            border: 1px solid #cbd5e1;
            border-radius: 999px;
            color: #334155;
            background: #f8fafc;
            padding: .25rem .65rem;
            font-size: .85rem;
        }
        .small-muted {
            color: #64748b;
            font-size: .9rem;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #d8e0ee;
            border-radius: 8px;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(payload: dict[str, Any]) -> None:
    source = payload.get("source", {})
    article_count = len(payload.get("articles", []))
    title = html.escape(source.get("title", "Code CIMA des assurances"))
    edition = html.escape(source.get("edition", "2019"))
    st.markdown(
        f"""
        <section class="cima-hero">
          <h1>Assistant Code CIMA des assurances</h1>
          <p>
            Recherche documentaire dans <strong>{title}</strong>, édition {edition}.
            Les réponses citent les articles et affichent les extraits utilisés.
          </p>
          <div class="cima-pill-row">
            <span class="cima-pill">{article_count} articles indexés</span>
            <span class="cima-pill">Références obligatoires</span>
            <span class="cima-pill">Réponse refusée si la confiance est faible</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(payload: dict[str, Any]) -> tuple[int, str | None]:
    source = payload.get("source", {})
    st.sidebar.title("Réglages")
    st.sidebar.caption(
        "Le moteur est extractif : il cherche dans les articles du Code CIMA fourni "
        "et affiche les sources utilisées."
    )
    source_count = st.sidebar.slider(
        "Nombre d'articles à afficher",
        min_value=1,
        max_value=5,
        value=DEFAULT_SOURCE_COUNT,
    )
    st.sidebar.divider()
    st.sidebar.markdown("**Consulter directement un article**")
    article_lookup = st.sidebar.text_input("Numéro d'article", placeholder="Ex. 13 ou 308-1")
    direct_prompt = None
    if st.sidebar.button("Ouvrir l'article", use_container_width=True):
        if article_lookup.strip():
            direct_prompt = f"Article {article_lookup.strip()}"

    st.sidebar.divider()
    st.sidebar.markdown("**Questions suggérées**")
    samples = [
        "Quelles mentions doivent figurer dans le contrat d'assurance ?",
        "Quand la prime d'assurance doit-elle être payée ?",
        "Quels véhicules sont concernés par l'assurance automobile obligatoire ?",
        "Quelle autorisation faut-il pour exercer comme courtier d'assurance ?",
    ]
    for index, sample in enumerate(samples):
        if st.sidebar.button(sample, key=f"sample-{index}", use_container_width=True):
            direct_prompt = sample

    st.sidebar.divider()
    st.sidebar.markdown("**Source**")
    st.sidebar.write(source.get("title", "Code CIMA"))
    st.sidebar.write(f"Édition : {source.get('edition', '2019')}")
    st.sidebar.write(
        f"Pages indexées : {source.get('page_start', '?')} à {source.get('page_end', '?')}"
    )
    st.sidebar.caption(
        "Pour tenir compte de textes postérieurs à 2019, régénérez l'index depuis un PDF à jour."
    )
    return source_count, direct_prompt


def ensure_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


def main() -> None:
    st.set_page_config(
        page_title="Assistant Code CIMA",
        page_icon="C",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    ensure_session_state()

    try:
        payload = load_article_payload()
    except Exception as exc:
        st.error(
            "Impossible de charger le Code CIMA. Vérifiez l'accès Internet ou générez "
            f"un index local avec scripts/build_code_cima_index.py. Détail : {exc}"
        )
        st.stop()

    source_count, direct_prompt = render_sidebar(payload)
    render_header(payload)

    left, right = st.columns([0.7, 0.3], vertical_alignment="center")
    with left:
        st.markdown(
            '<p class="small-muted">Posez une question ou indiquez un numéro d\'article. '
            "Le chatbot répond uniquement avec les éléments trouvés dans l'index.</p>",
            unsafe_allow_html=True,
        )
    with right:
        if st.button("Réinitialiser la conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_response(message["content"])

    prompt = direct_prompt or st.chat_input("Votre question sur le Code CIMA")
    if not prompt:
        return

    prompt = prompt.strip()
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    response = make_response(prompt, payload, source_count)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()


if __name__ == "__main__":
    main()
