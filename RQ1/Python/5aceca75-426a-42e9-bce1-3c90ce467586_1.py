from bitcoinrpc.authproxy import AuthServiceProxy, JSONRPCException

rpc_user = 'dein_rpc_benutzername'
rpc_password = 'dein_rpc_passwort'
rpc_port = 'port_des_nodes'  # Standardport ist 18443 für regtest

rpc_connection = AuthServiceProxy(f'http://{rpc_user}:{rpc_password}@127.0.0.1:{rpc_port}')

# Beispielaufrufe
try:
    info = rpc_connection.getinfo()
    balance = rpc_connection.getbalance()
    new_address = rpc_connection.getnewaddress()
    
    print("Node Info:", info)
    print("Balance:", balance)
    print("New Address:", new_address)

except JSONRPCException as e:
    print("Error:", e.error['message'])
