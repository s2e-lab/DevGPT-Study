@app.route('/oauth_initialization', methods=['GET'])
def oauth_initialization():
    # Extract parameters from the request
    client_id = request.args.get('client_id')
    client_domain = request.args.get('client_domain')
    authorization_url = request.args.get('authorization_url')
    token_url = request.args.get('token_url')
    openplugin_callback_url = request.args.get('openplugin_callback_url')
    authorization_content_type = request.args.get('authorization_content_type')

    # Store these parameters in the session
    session['client_id'] = client_id
    session['client_domain'] = client_domain
    session['authorization_url'] = authorization_url
    session['token_url'] = token_url
    session['openplugin_callback_url'] = openplugin_callback_url
    session['authorization_content_type'] = authorization_content_type

    # Initialize the client with the provided client_id
    client = WebApplicationClient(client_id)

    # Generate a unique state value for this request
    state = os.urandom(16).hex()
    session['state'] = state

    # Prepare the authorization request
    authorization_url, headers, _ = client.prepare_authorization_request(
        authorization_url=authorization_url,
        state=state,
        redirect_url=openplugin_callback_url
    )

    # Redirect the user to the authorization_url
    return redirect(authorization_url)
