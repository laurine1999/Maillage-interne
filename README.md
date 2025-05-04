# 🔗 Outil de Maillage Interne Sémantique

Cet outil SEO analyse automatiquement les pages de votre site et recommande des liens internes à insérer, basés sur la similarité sémantique.

## Fonctionnalités

- Upload d’un fichier CSV d’URLs à enrichir
- Extraction de contenu propre
- Analyse sémantique via Sentence-BERT
- Suggestions de liens sémantiquement proches (jusqu'à 5 par page)
- Ancre recommandée automatiquement
- Export CSV prêt à intégrer

## Démarrage local

```bash
pip install -r requirements.txt
streamlit run outil_maillage_seo.py
```

## Déploiement

Déployez-le en ligne gratuitement via [Streamlit Cloud](https://streamlit.io/cloud).
