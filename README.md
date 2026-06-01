# Assistant Code CIMA des assurances

Application Streamlit de recherche documentaire dans le Code des assurances des États membres de la CIMA, édition 2019.

L'application utilise `data/code_cima_articles.json` lorsqu'il est présent. Sinon, elle télécharge le PDF officiel CIMA 2019 depuis `https://cima-afrique.org/wp-content/uploads/2023/06/CODE-CIMA-2019.pdf` et reconstruit l'index en cache au premier lancement.

## Lancer l'application

```powershell
pip install -r requirements.txt
streamlit run chatbotassurances.py
```

## Générer un index local optionnel

Pour éviter le téléchargement au démarrage, vous pouvez générer un index local depuis `CODE-CIMA-2019.pdf` :

```powershell
python .\scripts\build_code_cima_index.py "C:\chemin\vers\CODE-CIMA-2019.pdf" --output .\data\code_cima_articles.json
```

Par défaut, le script indexe les pages 37 à 408 du PDF, soit le Code des assurances proprement dit.

## Garde-fous

- Les réponses sont extractives et affichent les articles sources.
- Une réponse est refusée si le score ou la couverture des termes importants est trop faible.
- Les références indiquent le numéro d'article, le titre et les pages du PDF indexé.
