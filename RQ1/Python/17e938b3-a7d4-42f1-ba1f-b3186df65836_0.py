from datetime import datetime
import pytz

utc_now = datetime.now(pytz.utc)
iso_timestamp = utc_now.isoformat()
print(iso_timestamp)
