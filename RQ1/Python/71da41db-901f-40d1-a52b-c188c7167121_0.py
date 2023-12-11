#!/usr/bin/env python3

import re
import json
import sys

def parse_data(data):
    postings = data.strip().split("\n\n")

    output = []

    for posting in postings:
        posting_dict = {}
        lines = posting.split("\n")

        posting_dict["Company"] = {
            "Full Name": lines[0].strip(),
            "Name": lines[2].strip(),
            "Hiring Status": lines[4].strip() if "recruiting" in lines[4] else "Not specified"
        }

        position_data = re.match(r'(.*) (?:Software Engineer|Engineer) -? ?(.*)?', lines[1].strip())
        posting_dict["Position"] = {
            "Title": "Software Engineer" if "Software Engineer" in lines[1] else "Engineer",
            "Level": position_data.group(1),
            "Type": position_data.group(2) if position_data.group(2) else "Not specified"
        }

        location_data = re.match(r'(.*?) \((.*)\)', lines[3].strip())
        posting_dict["Location"] = {
            "Country": location_data.group(1),
            "Type": location_data.group(2)
        }

        output.append(posting_dict)

    return json.dumps(output, indent=4)


data = sys.stdin.read()
print(parse_data(data))
