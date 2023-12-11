from django_crontab import cronjobs

@cronjobs.register
def reset_levels():
    profiles = Profile.objects.all()
    for profile in profiles:
        profile.level = 1
        profile.save()
