import os, json, time, requests, pandas as pd

DATA = os.path.join("data/interactions.csv")
OUT  = os.path.join("artifacts/movie_catalog.json")

API_BASE = "http://128.2.220.241:8080/movie/"   

SLEEP_BETWEEN_CALLS = 0.02
RETRY = 2
TIMEOUT = 3

def fetch_movie(movie_id: str):
    url = API_BASE + movie_id
    tries = 0
    while True:
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            else:
                return None
        except (requests.RequestException, ValueError):
            tries += 1
            if tries > RETRY:
                return None
            time.sleep(0.2)

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # load prior cache if exists
    catalog = {}
    if os.path.exists(OUT):
        with open(OUT, "r") as f:
            catalog = json.load(f)

    print("Reading interactions to collect movie ids...")
    # read only the movie_id column (faster, less RAM)
    mids = pd.read_csv(DATA, usecols=["movie_id"], dtype={"movie_id":"string"})
    unique_ids = mids["movie_id"].dropna().unique().tolist()
    print(f"Unique movies in interactions: {len(unique_ids)}")

    missing = [m for m in unique_ids if m not in catalog]
    print(f"To fetch: {len(missing)} (cached: {len(catalog)})")

    for i, mid in enumerate(missing, 1):
        js = fetch_movie(mid)
        if js is not None:
            # normalize to keep only the lightweight bits 
            catalog[mid] = {
                "genres": js.get("genres", []),
                "title": js.get("title", mid),
            }
        if i % 200 == 0:
            print(f"Fetched {i}/{len(missing)}; saving checkpoint...")
            with open(OUT, "w") as f:
                json.dump(catalog, f)
        time.sleep(SLEEP_BETWEEN_CALLS)

    with open(OUT, "w") as f:
        json.dump(catalog, f)
    print(f"Wrote {OUT} with {len(catalog)} movies")

if __name__ == "__main__":
    main()
