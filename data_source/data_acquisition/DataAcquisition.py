import requests
import feedparser
import time
import math
import json
from datetime import datetime

def fetch_arxiv_papers(search_query, total_results=10000, batch_size=100, sort_by='relevance', sort_order='descending', delay=3):
    base_url = 'http://export.arxiv.org/api/query?'
    max_batch_size = 1000
    if batch_size > max_batch_size:
        batch_size = max_batch_size

    num_batches = math.ceil(total_results / batch_size)
    papers = []

    for batch_num in range(num_batches):
        start = batch_num * batch_size
        params = {
            'search_query': f'all:{search_query}',
            'start': start,
            'max_results': min(batch_size, total_results - start),
            'sortBy': sort_by,
            'sortOrder': sort_order
        }

        r = requests.get(base_url, params=params)
        r.raise_for_status()
        feed = feedparser.parse(r.text)

        for entry in feed.entries:
            # find the PDF link
            pdf_url = next((lnk.href for lnk in entry.links 
                            if lnk.type == 'application/pdf'), None)
            if not pdf_url:
                continue

            papers.append({
                "title": entry.title.strip().replace('\n', ' '),
                "url": pdf_url
            })

        print(f"Batch {batch_num+1}/{num_batches} fetched, total so far: {len(papers)}")
        time.sleep(delay)

    # save to JSON
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"arxiv_{search_query.replace(' ', '_')}_{ts}.json"
    with open(fname, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    print(f"Done—{len(papers)} papers saved to {fname}")

if __name__ == "__main__":
    fetch_arxiv_papers(
        search_query="large language model",
        total_results=10000,
        batch_size=100,
        sort_by='relevance',
        sort_order='descending',
        delay=3
    )
