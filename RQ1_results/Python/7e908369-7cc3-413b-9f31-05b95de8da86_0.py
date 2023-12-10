@app.route("/auth/oauth_exchange", methods=['POST'])
def oauth_exchange():
    request_data = request.get_json(force=True)
    print(f"oauth_exchange {request_data=}")

    if request_data["client_id"] != OPENAI_CLIENT_ID:
        raise RuntimeError("bad client ID")
    if request_data["client_secret"] != OPENAI_CLIENT_SECRET:
        raise RuntimeError("bad client secret")
    if request_data["code"] != OPENAI_CODE:
        raise RuntimeError("bad code")

    return {
        "access_token": OPENAI_TOKEN,
        "token_type": "bearer"
    }, 200
