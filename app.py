from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression
import pickle
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords si ce n'est pas fait
nltk.download('stopwords')

app = Flask(__name__)

# Charger le modèle et le vectorizer
with open('./Model_tf_idf.pkl', 'rb') as f_model:
    model = pickle.load(f_model)

with open('./Vec_tf_idf.pkl', 'rb') as f_vectorizer:
    vectorizer = pickle.load(f_vectorizer)

# Fonction de clean et mise en forme
def Fonction_transfo_data(text):
    word_tokens_split = text.split(' ')
    stop_w = list(set(stopwords.words('english')))
    filtered_w = [w for w in word_tokens_split if not w in stop_w]
    lw = [w.lower() for w in filtered_w if not w.startswith("@")]
    filtered_w = [w for w in lw if not w in stop_w]
    data_final = ' '.join(filtered_w)
    return data_final

@app.route('/')
def home():
    return "Test TF IDF"

@app.route('/predict', methods=['GET,POST'])
def predict():
    if request.content_type != 'application/json':
        return jsonify({'error': 'Content-Type must be application/json'}), 400

    data = request.get_json()
    text = data.get("text")
    if not text:
        return jsonify({'error': 'Aucun texte fourni pour la prédiction'}), 400

    text_clean = Fonction_transfo_data(text)

    # Transforme le texte nettoyé avec le vectoriseur
    text_vectorized = vectorizer.transform([text_clean])

    # Utiliser predict_proba pour obtenir la probabilité de chaque classe
    prediction_proba = model.predict_proba(text_vectorized)

    # Obtenir la probabilité de la classe 1 (positif)
    sentiment_value = float(prediction_proba[0][1])  # Probabilité d'être "positif"
    sentiment = 'positif' if sentiment_value >= 0.5 else 'negatif'

    return jsonify({'sentiment_value': sentiment_value, 'sentiment': sentiment})

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(host = '0.0.0.0',port = 5000)
