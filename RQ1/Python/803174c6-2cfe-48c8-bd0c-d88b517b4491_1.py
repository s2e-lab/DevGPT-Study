import os
import google.oauth2.credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Set your credentials file path and video file path
CLIENT_SECRETS_FILE = 'path/to/client_secrets.json'
VIDEO_FILE = 'path/to/video.mp4'

# Create credentials
credentials = None
if os.path.exists('token.json'):
    credentials = google.oauth2.credentials.Credentials.from_authorized_user_file('token.json')
if not credentials or not credentials.valid:
    flow = google.auth.OAuth2FlowFromClientSecrets(CLIENT_SECRETS_FILE, scopes=['https://www.googleapis.com/auth/youtube.upload'])
    flow.run_local_server(port=8080, prompt='consent', authorization_prompt_message='')
    credentials = flow.credentials
    with open('token.json', 'w') as token:
        token.write(credentials.to_json())

# Build the YouTube API service
youtube = build('youtube', 'v3', credentials=credentials)

# Create a request to insert the video
request = youtube.videos().insert(
    part='snippet,status',
    body={
        'snippet': {
            'title': 'My Uploaded Video',
            'description': 'Description of my video',
            'tags': ['tag1', 'tag2'],
            'categoryId': '22'  # Category ID for 'People & Blogs'
        },
        'status': {
            'privacyStatus': 'private'  # 'private', 'public', 'unlisted'
        }
    },
    media_body=MediaFileUpload(VIDEO_FILE)
)

# Execute the request to upload the video
response = request.execute()
print(response)
