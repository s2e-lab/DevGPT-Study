import anvil.tables as tables
from anvil.tables import app_tables

def store_line_data(party, people, time_taken):
    app_tables.line_data.add_row(party=party, people=people, time_taken=time_taken)
    return "Data stored successfully."

def estimate_line_time(party, current_people):
    all_data = app_tables.line_data.search(party=party)
    total_people = 0
    total_time = 0
    
    for data in all_data:
        total_people += data['people']
        total_time += data['time_taken']
        
    if total_people == 0:
        return "Insufficient data for this party."
    
    avg_time_per_person = total_time / total_people
    estimated_time = avg_time_per_person * current_people
    
    return f"Estimated time: {estimated_time} minutes"
