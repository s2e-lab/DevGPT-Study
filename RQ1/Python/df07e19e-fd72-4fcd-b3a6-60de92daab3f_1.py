@tasks.loop(seconds=60)
async def check_reminders(self):
    now_unix = int(datetime.utcnow().timestamp())  # Converting current time to UNIX time
    for reminder in self.backend.filter(ReminderEntry, {}):  # Filtering manually as before
        reminder_time_unix = reminder['time']
        if reminder_time_unix <= now_unix:
            # The rest remains the same
