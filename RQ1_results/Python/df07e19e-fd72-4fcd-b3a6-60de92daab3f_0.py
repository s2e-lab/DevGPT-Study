reminder_time_unix = int(reminder_time.timestamp())  # Converting datetime to UNIX time
reminder = ReminderEntry(
    {
        "time": reminder_time_unix,  # Saving UNIX timestamp
        "text": text,
        "user_id": ctx.author.id,
        "channel_id": ctx.channel.id,
        "location": location,
        "nag": False,
    }
)
