import feedparser
import pandas as pd
from datetime import datetime


FEEDS = {
    "Pakistan": "https://www.dawn.com/feeds/home",
    "World": "https://www.dawn.com/feeds/world-news",
    "Business": "https://www.dawn.com/feeds/business-news",
    "Technology": "https://www.dawn.com/feeds/technology",
    "Sports": "https://www.dawn.com/feeds/sport",
}


def fetch_articles():
    articles = []

    for category, url in FEEDS.items():
        feed = feedparser.parse(url)

        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            published = entry.get("published", "")

            try:
                date = datetime(*entry.published_parsed[:6])
            except Exception:
                date = datetime.now()

            articles.append({
                "title": title,
                "content": summary,
                "category": category,
                "date": date,
            })

    df = pd.DataFrame(articles)
    df.drop_duplicates(subset="title", inplace=True)
    df.sort_values("date", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df
