def generate_abbreviated_highlight(actor_names, verb, object_, total_actors):
    if total_actors == 1:
        return f"{actor_names[0]} {verb} {object_}."
    elif total_actors == 2:
        return f"{actor_names[0]} and {actor_names[1]} {verb} {object_}."
    else:
        remaining_count = total_actors - 2
        return f"{actor_names[0]}, {actor_names[1]} and {remaining_count} others {verb} {object_}."

# Initialize a dictionary to store summarized activities
summarized_activities = {}

# Example list of activities (replace this with your actual activity data)
activities = [
    {"actor": "UserA", "verb": "liked", "object": "your post"},
    {"actor": "UserB", "verb": "liked", "object": "your post"},
    {"actor": "UserC", "verb": "liked", "object": "your post"},
    # Add more activities here
]

# Summarize the activities
for activity in activities:
    actor = activity["actor"]
    verb = activity["verb"]
    object_ = activity["object"]
    
    if (verb, object_) in summarized_activities:
        summarized_activities[(verb, object_)]["actors"].append(actor)
        summarized_activities[(verb, object_)]["count"] += 1
    else:
        summarized_activities[(verb, object_)] = {"actors": [actor], "count": 1}

# Generate and print the summarized highlights
for (verb, object_), data in summarized_activities.items():
    actor_names = data["actors"]
    total_actors = data["count"]
    highlight = generate_abbreviated_highlight(actor_names, verb, object_, total_actors)
    print(highlight)
