if newRequest["client_id"] != OPENAI_CLIENT_ID:
    return {"error": "bad client ID"}, 400
if newRequest["client_secret"] != OPENAI_CLIENT_SECRET:
    return {"error": "bad client secret"}, 400
if newRequest["code"] != OPENAI_CODE:
    return {"error": "bad code"}, 400
