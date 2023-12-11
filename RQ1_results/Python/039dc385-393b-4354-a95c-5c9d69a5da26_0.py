# Pseudo code
def crawl_stargazers(repository_id, last_highest_order):
    # Initialize star_order with last highest order plus some large number (e.g., 10000)
    star_order = last_highest_order + 10000

    while True:
        stargazers = fetch_stargazers_from_api(repository_id)
        
        if not stargazers:
            break

        for stargazer in stargazers:
            store_stargazer_data(repository_id, stargazer, star_order)
            star_order -= 1
