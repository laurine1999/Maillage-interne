
# outil_maillage_seo.py

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import trafilatura
from sentence_transformers import SentenceTransformer, util
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Chargement du modèle NLP
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Fonction pour nettoyer le texte HTML
def clean_html(html):
    downloaded = trafilatura.extract(html, include_comments=False, include_tables=False)
    return downloaded if downloaded else ''

# Fonction pour extraire texte d'une URL
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return clean_html(response.text)
        else:
            return ''
    except:
        return ''

# Fonction pour obtenir les liens internes d'une page
def extract_internal_links(url, domain):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            links = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                if domain in href or href.startswith('/'):
                    full_link = href if domain in href else domain + href
                    links.add(full_link.strip())
            return links
        else:
            return set()
    except:
        return set()

# Fonction pour générer les embeddings d'un texte
def get_embedding(text):
    return model.encode(text, convert_to_tensor=True)

# Fonction pour suggérer des liens sémantiques
def suggest_links(df, seuil, exclusion):
    embeddings = []
    contenus = []
    urls = df['URL'].tolist()
    domaines = [re.match(r'https?://[^/]+', u).group(0) for u in urls]

    for url in urls:
        text = extract_text_from_url(url)
        contenus.append(text)
        embeddings.append(get_embedding(text))

    suggestions = []
    for i, emb_source in enumerate(embeddings):
        url_source = urls[i]
        if any(excl in url_source for excl in exclusion):
            continue
        liens_existants = extract_internal_links(url_source, domaines[i])
        scored_links = []
        for j, emb_cible in enumerate(embeddings):
            if i == j:
                continue
            score = float(util.cos_sim(emb_source, emb_cible))
            if score >= seuil and urls[j] not in liens_existants:
                scored_links.append((urls[j], score, generate_anchor_from_text(contenus[j])))
        # Trier et limiter à 5 suggestions par page
        scored_links = sorted(scored_links, key=lambda x: x[1], reverse=True)[:5]
        for url_cible, score, ancre in scored_links:
            suggestions.append({
                'URL source': url_source,
                'URL cible': url_cible,
                'Score de similarité': round(score, 3),
                'Ancre suggérée': ancre
            })
    return pd.DataFrame(suggestions)

# Fonction pour générer une ancre possible à partir du texte de la page cible
def generate_anchor_from_text(text, max_words=5):
    try:
        vectorizer = TfidfVectorizer(max_df=0.8, stop_words='french')
        X = vectorizer.fit_transform([text])
        scores = zip(vectorizer.get_feature_names_out(), X.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return ' '.join([w for w, s in sorted_scores[:max_words]])
    except:
        return 'Voir page liée'

# --- INTERFACE STREAMLIT ---
st.title("🔗 Outil de Maillage Interne Sémantique")

st.markdown("""
Cet outil analyse le contenu de vos pages et propose automatiquement des **liens internes à insérer** entre pages proches sémantiquement.

- Téléchargez un fichier CSV contenant une colonne `URL` avec les pages à enrichir.
- L’outil ne proposera **que des liens qui n’existent pas déjà** dans le code HTML.
- Export final au format CSV avec suggestions de liens et ancres.
""")

uploaded_file = st.file_uploader("📥 Charger votre fichier CSV avec les URLs", type=["csv"])

if uploaded_file:
    df_urls = pd.read_csv(uploaded_file)
    st.success(f"✅ {len(df_urls)} URLs chargées.")

    seuil = st.slider("Seuil de similarité minimale (0.9 recommandé)", 0.7, 1.0, 0.9, 0.01)
    exclusion_input = st.text_area("Exclure les URLs contenant ces mots (séparés par des virgules)", value="contact,mentions")
    exclusion_terms = [e.strip() for e in exclusion_input.split(',') if e.strip() != '']

    if st.button("🚀 Lancer l'analyse sémantique"):
        with st.spinner("Traitement en cours... Cela peut prendre quelques minutes."):
            results = suggest_links(df_urls, seuil, exclusion_terms)
        st.success("Analyse terminée ✅")

        st.dataframe(results)

        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📤 Télécharger les suggestions en CSV",
            data=csv,
            file_name='liens_semantiques_recommandes.csv',
            mime='text/csv'
        )
