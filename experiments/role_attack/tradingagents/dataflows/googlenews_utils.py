from datetime import datetime, timedelta
from gnews import GNews


def _parse_date(date_str):
    """Parse date string to datetime object."""
    if "/" in date_str:
        return datetime.strptime(date_str, "%m/%d/%Y")
    else:
        return datetime.strptime(date_str, "%Y-%m-%d")


def _format_date(raw_date):
    """Convert gnews date format to yyyy-mm-dd."""
    if not raw_date:
        return ""
    try:
        # gnews 날짜 형식: "Mon, 01 Jan 2024 12:00:00 GMT"
        dt = datetime.strptime(raw_date, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        try:
            dt = datetime.strptime(raw_date, "%a, %d %b %Y %H:%M:%S")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return raw_date


def _fetch_single_day(google_news, query, date_dt, fetch_full_article=True):
    """Fetch news for a single day."""
    next_day = date_dt + timedelta(days=1)
    google_news.start_date = (date_dt.year, date_dt.month, date_dt.day)
    google_news.end_date = (next_day.year, next_day.month, next_day.day)

    results = []
    try:
        articles = google_news.get_news(query)
        for article in articles:
            url = article.get("url", "")

            text = ""
            if fetch_full_article:
                try:
                    full_article = google_news.get_full_article(url)
                    if full_article:
                        text = full_article.text or ""
                except Exception:
                    pass

            results.append({
                "link": url,
                "title": article.get("title", ""),
                "snippet": article.get("description", ""),
                "text": text,
                "date": _format_date(article.get("published date", "")),
                "source": article.get("publisher", {}).get("title", "Unknown"),
            })
    except Exception as e:
        print(f"    Error on {date_dt.strftime('%Y-%m-%d')}: {e}")

    return results


def getNewsData(query, start_date, end_date, fetch_full_article=True):
    """
    Get Google News search results for a given query and date range using gnews library.
    Fetches news day by day to avoid the 100 results limit.

    query: str - search query
    start_date: str - start date in the format yyyy-mm-dd or mm/dd/yyyy
    end_date: str - end date in the format yyyy-mm-dd or mm/dd/yyyy
    fetch_full_article: bool - whether to fetch full article text (slower)
    """
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    google_news = GNews(
        language='en',
        country='US',
        max_results=100
    )

    news_results = []
    seen_links = set()  # 중복 제거용

    total_days = (end_dt - start_dt).days + 1
    current_dt = start_dt

    print(f"  Fetching {total_days} days of news...")

    day_count = 0
    while current_dt <= end_dt:
        day_count += 1
        day_results = _fetch_single_day(google_news, query, current_dt, fetch_full_article)

        # 중복 제거하며 추가
        for article in day_results:
            if article["link"] not in seen_links:
                seen_links.add(article["link"])
                news_results.append(article)

        if day_count % 7 == 0:
            print(f"    [{day_count}/{total_days}] days processed, {len(news_results)} articles so far")

        current_dt += timedelta(days=1)

    print(f"  Done: {len(news_results)} total articles")
    return news_results
