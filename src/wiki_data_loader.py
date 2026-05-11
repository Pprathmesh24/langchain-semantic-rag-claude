import wikipedia
import os

# List of companies to fetch
companies = {
    "Google": "Google",
    "Microsoft": "Microsoft",
    "Tesla": "Tesla, Inc.",
    "SpaceX": "SpaceX",
    "Nvidia": "Nvidia"
}

os.makedirs("rag_data", exist_ok=True)

for label, title in companies.items():
    try:
        page = wikipedia.page(title, auto_suggest=False)
        filename = f"rag_data/{label}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Title: {page.title}\n")
            f.write(f"URL: {page.url}\n")
            f.write(f"Summary:\n{page.summary}\n\n")
            f.write(f"Full Content:\n{page.content}")
        print(f"✅ Saved: {filename}")
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"⚠️ Disambiguation for {title}: {e.options[:5]}")
    except wikipedia.exceptions.PageError:
        print(f"❌ Page not found: {title}")