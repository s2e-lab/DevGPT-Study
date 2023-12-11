@app.route('/oauth_token', methods=['GET'])
def oauth_token():
    try:
        # Extract the state and code parameters
        state = request.args.get('state')
        code = request.args.get('code')

        # Retrieve the session using the state
        session_data = session.get(state)
        if not session_data:
            return jsonify({"error": "Invalid state"}), 400

        # Fetch the item from the 'openplugin-auth' collection using the client_domain
        item = db["openplugin-auth"].find_one({"domain": session_data["client_domain"]})
        if not item:
            return jsonify({"error": "Item not found"}), 404

        # Retrieve the client_secret from the item
        client_secret = item.get("oauth", {}).get("client_secret")
        if not client_secret:
            return jsonify({"error": "Client secret not found"}), 404

        # Initialize the client with the provided client_id
        client = WebApplicationClient(session_data["client_id"])

        # Prepare the token request
        token_request_headers = {
            "Content-Type": session_data["authorization_content_type"]
        }
        token = client.prepare_token_request(
            session_data["token_url"],
            authorization_response=request.url,
            redirect_url=f"{request.url_root.rstrip('/')}/oauth_token",
            client_id=session_data["client_id"],
            client_secret=client_secret
        )
        token_url, headers, data_string = token

        # Conditional handling based on content type
        data_dict = dict([pair.split('=') for pair in data_string.split('&')])
        if token_request_headers["Content-Type"] == "application/x-www-form-urlencoded":
            token_data = urllib.parse.urlencode(data_dict)
        elif token_request_headers["Content-Type"] == "application/json":
            token_data = json.dumps(data_dict)

        # Make the POST request to the token_url
        token_response = requests.post(
            token_url,
            headers=token_request_headers,
            data=token_data
        )

        # Parse the response data
        client.parse_request_body_response(json.dumps(token_response.json()))

        # Construct the redirect URL with the response data and other parameters
        params = {
            **token_response.json(),
            "client_domain": session_data["client_domain"],
            "oauth_token": "true"
        }
        redirect_url = f"{session_data['openplugin_callback_url']}?{urllib.parse.urlencode(params)}"

        return redirect(redirect_url)

    except Exception as e:
        error_class = type(e).__name__
        error_message = str(e)
        return jsonify({"error": f"{error_class} error: {error_message}"}), 500
