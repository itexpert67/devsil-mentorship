import re
import string
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "will", "would", "could", "should", "may", "might",
    "it", "its", "this", "that", "these", "those", "he", "she", "they",
    "we", "you", "i", "said", "also", "as", "his", "her", "their", "our",
    "not", "no", "after", "before", "about", "over", "under", "more", "than",
    "up", "out", "into", "through", "during", "than", "then", "so", "if",
    "while", "which", "who", "whom", "when", "where", "how", "what", "dawn",
    "pakistan", "s", "p", "new", "one", "two", "three", "year", "years",
    "says", "say", "according", "told", "added", "however", "reuters", "afp",
}

try:
    nlp_model = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity
    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, round(polarity, 4), round(subjectivity, 4)


def extract_entities(texts):
    if not SPACY_AVAILABLE:
        return {}, {}, {}

    people = {}
    places = {}
    orgs = {}

    for text in texts:
        doc = nlp_model(text[:1000])
        for ent in doc.ents:
            name = ent.text.strip()
            if len(name) < 3:
                continue
            if ent.label_ == "PERSON":
                people[name] = people.get(name, 0) + 1
            elif ent.label_ in ("GPE", "LOC"):
                places[name] = places.get(name, 0) + 1
            elif ent.label_ == "ORG":
                orgs[name] = orgs.get(name, 0) + 1

    people = dict(sorted(people.items(), key=lambda x: x[1], reverse=True)[:20])
    places = dict(sorted(places.items(), key=lambda x: x[1], reverse=True)[:20])
    orgs = dict(sorted(orgs.items(), key=lambda x: x[1], reverse=True)[:20])

    return people, places, orgs


def get_trending_keywords(texts, top_n=30):
    cleaned = [clean_text(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    try:
        tfidf = vectorizer.fit_transform(cleaned)
        scores = tfidf.sum(axis=0).A1
        names = vectorizer.get_feature_names_out()
        ranked = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
        return [(w, round(s, 3)) for w, s in ranked[:top_n]]
    except Exception:
        return []


def run_lda(texts, n_topics=6, n_words=8):
    cleaned = [clean_text(t) for t in texts]
    vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000)
    try:
        dtm = vectorizer.fit_transform(cleaned)
    except Exception:
        return {}, []

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)

    feature_names = vectorizer.get_feature_names_out()
    topics = {}

    for i, component in enumerate(lda.components_):
        top_indices = component.argsort()[-n_words:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        topics[f"Topic {i + 1}"] = top_words

    doc_topics = lda.transform(dtm)
    dominant = [f"Topic {t + 1}" for t in doc_topics.argmax(axis=1)]

    return topics, dominant


def detect_breaking_stories(df, window_hours=6, threshold=3):
    recent = df[df["date"] >= df["date"].max() - __import__("pandas").Timedelta(hours=window_hours)]
    if recent.empty:
        return []

    texts = (recent["title"] + " " + recent["content"]).tolist()
    keywords = get_trending_keywords(texts, top_n=10)
    breaking = []

    for word, score in keywords:
        count = sum(1 for t in texts if word in t.lower())
        if count >= threshold:
            breaking.append({"keyword": word, "mentions": count, "score": score})

    return breaking


def analyze(df):
    sentiments = []
    polarities = []
    subjectivities = []

    for _, row in df.iterrows():
        text = row["title"] + " " + row["content"]
        label, polarity, subjectivity = get_sentiment(text)
        sentiments.append(label)
        polarities.append(polarity)
        subjectivities.append(subjectivity)

    df["sentiment"] = sentiments
    df["polarity"] = polarities
    df["subjectivity"] = subjectivities

    texts = (df["title"] + " " + df["content"]).tolist()

    if len(texts) >= 5:
        topics, dominant = run_lda(texts)
        df["topic"] = dominant
    else:
        topics = {}
        df["topic"] = "N/A"

    people, places, orgs = extract_entities(texts)
    trending = get_trending_keywords(texts)
    breaking = detect_breaking_stories(df)

    return df, topics, people, places, orgs, trending, breaking
