
from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use('Agg')  # For non-GUI environments
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from django.http import JsonResponse
from django.core.files.storage import default_storage
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans

# sentiment analysis model
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_labels = ['negative', 'neutral', 'positive']

is_model_loaded = False

sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer, device=-1)
is_sentiment_model_loaded = True

#--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==--==
def home(request):
    if not is_sentiment_model_loaded:
        return render(request, "loading.html")
    return render(request, "upload.html", {
        'sentiment_results': [],
        'topics': []
    })

#==--==--Sentiment Analysis View--==--==--==--==--==--==--==--==--==--==--==--==

def truncate_text(text):
    tokens = sentiment_tokenizer.tokenize(str(text))
    if len(tokens) > 512:
        tokens = tokens[:512]
        return sentiment_tokenizer.convert_tokens_to_string(tokens)
    return text

def analyze_sentiment_view(request):
    sentiment_results = []
    pie_chart = None
    report = None

    if request.method == "POST" and request.POST.get("action") == "sentiment":
        file = request.FILES.get("csv_file_sentiment")
        text_column = request.POST.get("text_column")
        score_column = request.POST.get("score_column")
        model_choice = request.POST.get("model_choice", "transformer")
        if file and text_column:
            file_path = default_storage.save(file.name, file)
            df = pd.read_csv(file_path)

            # Validate text column
            if text_column not in df.columns:
                return HttpResponse(f"Text column '{text_column}' not found in CSV.", status=400)

            df[text_column] = df[text_column].fillna("")
            df = df[df[text_column].apply(lambda x: isinstance(x, str))]
            df["reviewText_trunc"] = df[text_column].apply(truncate_text)
            df = df[df["reviewText_trunc"].apply(lambda x: isinstance(x, str) and x.strip() != "")]

            has_score = score_column and score_column in df.columns

            if has_score:
                df = df[pd.to_numeric(df[score_column], errors="coerce").notnull()]
                df[score_column] = df[score_column].astype(float)
                def map_sentiment(score):
                    if score <= 2:
                        return "negative"
                    elif score == 3:
                        return "neutral"
                    else:
                        return "positive"
                df["true_sentiment"] = df[score_column].apply(map_sentiment)

            texts = df["reviewText_trunc"].tolist()
            texts = [text for text in texts if isinstance(text, str) and text.strip() != ""]

            if texts:
                try:
                    if model_choice == "naive_bayes" and has_score:
                        # Naive Bayes (supervised)
                        vectorizer = TfidfVectorizer()
                        X = vectorizer.fit_transform(texts)
                        y = df["true_sentiment"]
                        clf = MultinomialNB()
                        clf.fit(X, y)
                        y_pred = clf.predict(X)
                        df["predicted_sentiment"] = y_pred
                    elif model_choice == "clustering":
                        # Clustering (unsupervised)
                        vectorizer = TfidfVectorizer()
                        X = vectorizer.fit_transform(texts)
                        n_clusters = 3
                        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                        clusters = kmeans.fit_predict(X)
                        cluster_labels = {i: sentiment_labels[i] for i in range(n_clusters)}
                        df["predicted_sentiment"] = [cluster_labels[c] for c in clusters]
                    else:
                        # Default: transformer
                        predictions = sentiment_pipeline(texts, truncation=True, max_length=512)
                        label_map = {
                            "LABEL_0": "negative",
                            "LABEL_1": "neutral",
                            "LABEL_2": "positive"
                        }
                        df = df.iloc[:len(predictions)]
                        df["predicted_sentiment"] = [label_map.get(p["label"], p["label"]) for p in predictions]
                        df["predicted_sentiment"] = df["predicted_sentiment"].str.lower().str.strip()

                    # Pie chart always works
                    sentiment_counts = df["predicted_sentiment"].value_counts().reindex(sentiment_labels, fill_value=0)
                    fig, ax = plt.subplots()
                    ax.pie(sentiment_counts, labels=sentiment_labels, autopct='%1.1f%%')
                    ax.set_title("Predicted Sentiment Distribution")
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    pie_chart = base64.b64encode(buffer.read()).decode('utf-8')
                    buffer.close()

                    sentiment_results = df["predicted_sentiment"].tolist()

                    # Only calculate classification report if score_column is provided and valid
                    if has_score:
                        df["true_sentiment"] = df["true_sentiment"].str.lower().str.strip()
                        df = df[df["true_sentiment"].notna() & df["predicted_sentiment"].notna()]
                        y_true = df["true_sentiment"]
                        y_pred = df["predicted_sentiment"]
                        report = classification_report(y_true, y_pred, output_dict=False)
                    else:
                        report = "Classification report not available (no score column provided, for testing the data)."
                except Exception as e:
                    return HttpResponse(f"Pipeline error: {e}", status=500)
            else:
                report = "No valid reviews to analyze."

    return render(request, "upload.html", {
        "sentiment_results": sentiment_results,
        "sentiment_chart": pie_chart,
        "classification_report": report
    })
#==--==--==--==--==--==-Topic Modeling--==--==--==--==--==--==--==--==
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

def topic_modeling_view(request):
    topic_words = []
    error = None
    if request.method == "POST" and request.POST.get("action") == "topic":
        print("Processing topic modeling request")
        file = request.FILES.get("csv_file_topic")
        text_column = request.POST.get("topic_text_column")
        if file and text_column:
            file_path = default_storage.save(file.name, file)
            df = pd.read_csv(file_path)
            if text_column not in df.columns:
                return HttpResponse(f"Text column '{text_column}' not found in CSV.", status=400)
            texts = df[text_column].fillna("").astype(str).tolist()
            # Remove empty or whitespace-only texts
            texts = [t for t in texts if t.strip() != ""]
            if not texts:
                error = "No valid texts found in the selected column."
            else:
                try:
                    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                    X = vectorizer.fit_transform(texts)
                    if X.shape[1] == 0:
                        error = "No valid vocabulary found. Try different text column or check your data."
                    else:
                        n_topics = 5
                        nmf = NMF(n_components=n_topics, random_state=1)
                        W = nmf.fit_transform(X)
                        H = nmf.components_
                        feature_names = vectorizer.get_feature_names_out()
                        for topic_idx, topic in enumerate(H):
                            top_words = [feature_names[i] for i in topic.argsort()[:-8:-1]]
                            topic_label = ", ".join(top_words)  # This acts as the topic "name"
                            topic_words.append({"label": topic_label, "words": top_words})
                except ValueError as ve:
                    error = f"Topic modeling error: {ve}"
    return render(request, "upload.html", {
        "topic_words": topic_words,
        "topic_error": error
    })
#==--==--==--==--==--==-End of Topic Modeling--==--==--==--==--==--==--==--==
import wikipedia

def scrape_wikipedia_view(request):
    wiki_content = ""
    wiki_error = None
    if request.method == "POST" and request.POST.get("action") == "scrape_wikipedia":
        query = request.POST.get("wiki_query")
        if not query:
            wiki_error = "Please enter a Wikipedia search term."
        else:
            try:
                # Get the summary (first paragraph) of the Wikipedia page
                wiki_content = wikipedia.summary(query, sentences=5)
            except wikipedia.DisambiguationError as e:
                wiki_error = f"Disambiguation error: {e.options[:5]} (please be more specific)"
            except wikipedia.PageError:
                wiki_error = "No Wikipedia page found for your query."
            except Exception as e:
                wiki_error = f"Error: {e}"
    return render(request, "upload.html", {
        "wiki_content": wiki_content,
        "wiki_error": wiki_error,
    })