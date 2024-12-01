import nltk
import streamlit as st
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Télécharger les ressources nécessaires
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Chargement du fichier texte contenant les questions et réponses
def load_faq(file_path):
    questions = []
    answers = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            question = lines[i].replace('Question:', '').strip()
            if i + 1 < len(lines):
                answer = lines[i + 1].replace('Réponse:', '').strip()
            else:
                answer = "Réponse non disponible."
            questions.append(question)
            answers.append(answer)
    return questions, answers

# Prétraitement des phrases
def preprocess(text):
    # Tokenisation en phrases
    sentences = sent_tokenize(text, language='french')
    # Tokenisation en mots pour chaque phrase
    words = [word_tokenize(sentence, language='french') for sentence in sentences]
    # Aplatir la liste de listes
    words = [word for sublist in words for word in sublist]
    # Suppression des mots vides et de la ponctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('french') and word not in string.punctuation]
    # Lemmatisation des mots
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Fonction pour obtenir la réponse la plus pertinente
def get_most_relevant_response(user_query, vectorizer, tfidf_matrix, questions, answers):
    user_query_processed = preprocess(user_query)
    user_query_vector = vectorizer.transform([user_query_processed])
    cosine_similarities = cosine_similarity(user_query_vector, tfidf_matrix)
    most_similar_index = np.argmax(cosine_similarities)
    return questions[most_similar_index], answers[most_similar_index]

# Fonction principale du chatbot
def main():
    st.title("Chatbot Assurance Automobile en Côte d'Ivoire")
    st.write("Posez une question concernant la souscription d'une assurance pour votre véhicule personnel en Côte d'Ivoire.")

    # Charger les questions et réponses
    questions, answers = load_faq('assurance_automobile_faq.txt')

    # Prétraiter les questions
    questions_preprocessed = [preprocess(question) for question in questions]

    # Créer le TF-IDF vectorizer et ajuster sur les questions prétraitées
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions_preprocessed)

    # Champ de saisie pour la question de l'utilisateur
    user_query = st.text_input("Votre question :")

    # Bouton pour soumettre la question
    if st.button("Soumettre"):
        if user_query:
            question, response = get_most_relevant_response(user_query, vectorizer, tfidf_matrix, questions, answers)
            st.write(f"**Question posée :** {question}")
            st.markdown(f'<div style="background-color: #FFECB3; padding: 10px; border-radius: 5px;">**Réponse :** {response}</div>', unsafe_allow_html=True)

            # Demander la satisfaction de l'utilisateur
            st.write("Êtes-vous satisfait de la réponse ?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("OUI", key="satisfait"):
                    st.write("Merci pour votre retour positif !")
            with col2:
                if st.button("NON", key="non_satisfait"):
                    st.write("Nous sommes désolés que la réponse ne vous ait pas satisfait.")

            # Proposer les actions suivantes
            st.markdown("**<span style='color:red;'>Que souhaitez-vous faire ensuite ?</span>**", unsafe_allow_html=True)
            col3, col4 = st.columns(2)
            with col3:
                if st.button("Poser une autre question", key="autre_question"):
                    st.experimental_rerun()
            with col4:
                if st.button("Quitter", key="quitter"):
                    st.markdown('<div style="background-color: #E1BEE7; padding: 10px; border-radius: 5px;">Merci de nous avoir consultés. À très bientôt !</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
