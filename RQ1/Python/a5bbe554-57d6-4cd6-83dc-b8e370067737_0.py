def print_hosts_info(hosts):
    print("IP Address,MAC Address,Host Name,Manufacturer")  # Header row
    for host in hosts:
        mac = get_mac(host)
        host_name = get_hostname(host)
        try:
            manufacturer = get_manufacturer(mac)
        except ValueError:
            manufacturer = 'Unknown'
        print(f'{host},{mac},{host_name},{manufacturer}')  # Data rows
