import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Slackボットのトークンを環境変数から取得
slack_token = os.environ["SLACK_API_TOKEN"]
client = WebClient(token=slack_token)

def send_message_to_channels(message):
    try:
        # チャンネル一覧を取得
        channels = client.conversations_list(types="public_channel,private_channel")["channels"]

        for channel in channels:
            channel_id = channel["id"]
            channel_name = channel["name"]

            try:
                # メッセージを送信
                response = client.chat_postMessage(channel=channel_id, text=message)
                print(f"Sent message to #{channel_name}: {response['ts']}")
            except SlackApiError as e:
                print(f"Failed to send message to #{channel_name}: {e.response['error']}")
    
    except SlackApiError as e:
        print(f"Failed to fetch channel list: {e.response['error']}")

# メッセージを送信する
send_message_to_channels("Hello, Slack channels!")
