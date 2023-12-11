from datetime import datetime
import pytz

# Events data
events = [
    {
        "summary": "Start building the foundation of your portfolio website",
        "start": datetime(2023, 6, 20, 10, 30),
        "end": datetime(2023, 6, 20, 11, 30),
        "description": "Since this is a long-term project, an hour of focused work can help you make progress."
    },
    # ... Add the rest of the events in the same format as above
]

# Calendar content
content = [
    "BEGIN:VCALENDAR",
    "VERSION:2.0",
    "PRODID:-//ChatGPT//EN",
]

# Add events to the content
for event_data in events:
    content.append("BEGIN:VEVENT")
    content.append(f"SUMMARY:{event_data['summary']}")
    content.append(f"DTSTART;TZID=US/Eastern:{event_data['start'].strftime('%Y%m%dT%H%M%S')}")
    content.append(f"DTEND;TZID=US/Eastern:{event_data['end'].strftime('%Y%m%dT%H%M%S')}")
    content.append(f"DESCRIPTION:{event_data['description']}")
    content.append("END:VEVENT")

# End of calendar content
content.append("END:VCALENDAR")

# Joining content and saving to .ics file
ics_content = "\r\n".join(content)
ics_file_path = 'schedule.ics'
with open(ics_file_path, 'w') as f:
    f.write(ics_content)

print(f".ics file created at {ics_file_path}")
