import requests
import json

# API endpoints
get_url = "https://openlibrary.org{work_key}/editions.json"
put_url = "https://openlibrary.org{ol_key}.json"
keys_url = "https://github.com/internetarchive/openlibrary/files/11679407/works-null-lccn.txt"

# Dry run option
dry_run = True  # Set to False to actually make changes

# Fetch the work keys from the URL
response = requests.get(keys_url)
response.raise_for_status()  # Raise an exception if the request failed

work_keys = response.text.strip().split('\n')

for work_key in work_keys:
    # fetch the list of editions for the current work_key
    response = requests.get(get_url.format(work_key=work_key))

    if response.status_code == 200:
        data = response.json()

        for entry in data['entries']:
            if 'lccn' in entry and entry['lccn'] == [None]:
                # If dry_run is True, just print the changes
                if dry_run:
                    print(f"Would remove lccn from {entry['key']}")
                else:
                    # remove the lccn field
                    del entry['lccn']

                    # update the edition record
                    update_response = requests.put(put_url.format(ol_key=entry['key']), json=entry)

                    if update_response.status_code != 200:
                        print(f"Failed to update {entry['key']}")
    else:
        print(f"Failed to get editions for {work_key}")
