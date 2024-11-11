import ipaddress


class IPGenerator:
    def __init__(self, network_address):
        self.network = ipaddress.ip_network(network_address)
        self.host_iter = iter(self.network.hosts())
        self.devices_ip = {}

    def next_host_ip(self, host_name):
        try:
            ip = str(next(self.host_iter))
            self.devices_ip[host_name] = ip
            print(ip)
            return ip
        except StopIteration:
            raise ValueError("No more available IP addresses in the network")

    def get_host_ip(self, host_name):
        return self.devices_ip.get(host_name)


if __name__ == '__main__':
    # Example usage:
    generator = IPGenerator("192.168.1.0/24")
    print(generator.next_host_ip('host1'))  # Output: 192.168.1.1
    print(generator.next_host_ip('host2'))  # Output: 192.168.1.2
    print(generator.get_host_ip('host1'))  # Output: 192.168.1.1
    print(generator.get_host_ip('host3'))  # Output: None
    generator = IPGenerator("2001:db8::/32")
    print(generator.next_host_ip('sensor1'))  # Output: 2001:db8::1
    print(generator.next_host_ip('sensor2'))  # Output: 2001:db8::2
    print(generator.get_host_ip('sensor1'))  # Output: 2001:db8::1
