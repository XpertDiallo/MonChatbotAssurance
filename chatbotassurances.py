import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Téléchargement des ressources nécessaires de NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Nécessaire pour certaines versions de NLTK

# Chargement du fichier texte contenant les questions et réponses
with open('assurance_automobile_faq.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]  # Supprime les lignes vides et les espaces inutiles

# Initialisation des listes pour stocker les questions et réponses
questions = []
reponses = []

# Parcours des lignes pour extraire les questions et réponses
i = 0
while i < len(lines):
    if lines[i].startswith('Question:'):
        question = lines[i].replace('Question:', '').strip()
        if i + 1 < len(lines) and lines[i + 1].startswith('Réponse:'):
            reponse = lines[i + 1].replace('Réponse:', '').strip()
            questions.append(question)
            reponses.append(reponse)
            i += 2  # Passe à la paire suivante
        else:
            i += 1  # Passe à la ligne suivante si la réponse est manquante
    else:
        i += 1  # Passe à la ligne suivante si la question est manquante

# Vérifie que nous avons bien des paires de questions et réponses
assert len(questions) == len(reponses), "Le nombre de questions et de réponses ne correspond pas."

# Définition d'une fonction pour prétraiter le texte
def preprocess(text):
    # Tokenisation en mots
    words = word_tokenize(text, language='french')
    # Suppression des mots vides et de la ponctuation
    words = [word.lower() for word in words if word.lower() not in stopwords.words('french') and word not in string.punctuation]
    # Lemmatisation des mots
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# Prétraitement des questions
questions_preprocessed = [preprocess(question) for question in questions]

# Fonction pour obtenir la réponse la plus pertinente
def get_most_relevant_response(user_query):
    user_query_processed = preprocess(user_query)
    vectorizer = TfidfVectorizer().fit(questions_preprocessed + [user_query_processed])
    vectors = vectorizer.transform(questions_preprocessed + [user_query_processed])
    user_vector = vectors[-1]
    cosine_similarities = cosine_similarity(user_vector, vectors[:-1])
    most_similar_index = np.argmax(cosine_similarities)
    return reponses[most_similar_index]

# Fonction principale du chatbot
def main():
    # CSS pour la personnalisation des boutons et de la zone de réponse
    st.markdown("""
        <style>
        .stButton > button {
            background-color: #007BFF;
            color: white;
        }
        .stButton > button.red-button {
            background-color: #DC3545;
            color: white;
        }
        .response-box {
            background-color: #FFF3CD;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #FFEEBA;
        }
        .farewell-message {
            background-color: #6F42C1;
            color: white;
            padding: 10px;
            border-radius: 5px;
            border: 1px solid #5A32A3;
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)

    st.title("Chatbot Assurance Automobile en Côte d'Ivoire")
    st.write("Posez une question concernant la souscription d'une assurance pour votre véhicule personnel en Côte d'Ivoire.")
    
    if 'response' not in st.session_state:
        st.session_state.response = None
    if 'satisfaction' not in st.session_state:
        st.session_state.satisfaction = None

    with st.form(key='question_form', clear_on_submit=True):
        user_query = st.text_input("Votre question :")
        submit_button = st.form_submit_button(label='Soumettre')
    
    if submit_button and user_query:
        st.session_state.response = get_most_relevant_response(user_query)
        st.session_state.satisfaction = None
        st.write(f"**Question :** {user_query}")
        st.markdown(f'<div class="response-box">**Réponse :** {st.session_state.response}</div>', unsafe_allow_html=True)

    if st.session_state.response and st.session_state.satisfaction is None:
        st.markdown("**:blue[Êtes-vous satisfait de la réponse ?]**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button('OUI'):
                st.session_state.satisfaction = 'oui'
        with col2:
            if st.button('NON', key='non_button', args=('red-button',)):
                st.session_state.satisfaction = 'non'
        
    if st.session_state.satisfaction:
        if st.session_state.satisfaction == 'oui':
            st.write("Merci pour votre retour positif !")
        else:
            st.write("Nous sommes désolés que la réponse ne vous ait pas satisfait.")
        
        st.markdown("**:red[Que souhaitez-vous faire ensuite ?]**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Poser une autre question'):
                st.session_state.response = None
                st.session_state.satisfaction = None
        with col2:
            if st.button('Quitter', key='quitter_button', args=('red-button',)):
                st.markdown('<div class="farewell-message">Merci de nous avoir consultés. À très bientôt !</div>', unsafe_allow_html=True)
                st.stop()

if __name__ == "__main__":
    main()
