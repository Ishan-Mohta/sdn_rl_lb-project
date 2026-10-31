"""
Mininet Topology for SDN Load Balancing.
Run using: sudo python mininet_topology.py
"""
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel

class LoadBalancerTopo(Topo):
    def build(self):
        # Add switch
        s1 = self.addSwitch('s1')

        # Add 3 servers with different capacities (simulated via CPU limits in a real setup)
        h1 = self.addHost('h1', ip='10.0.0.1') # Server 1
        h2 = self.addHost('h2', ip='10.0.0.2') # Server 2
        h3 = self.addHost('h3', ip='10.0.0.3') # Server 3
        
        # Add 1 client
        c1 = self.addHost('c1', ip='10.0.0.100')

        # Connect to switch
        self.addLink(h1, s1, port1=1, port2=1)
        self.addLink(h2, s1, port1=1, port2=2)
        self.addLink(h3, s1, port1=1, port2=3)
        self.addLink(c1, s1, port1=1, port2=4)

def run():
    topo = LoadBalancerTopo()
    # Connect to the Ryu controller running on localhost:6653
    net = Mininet(topo=topo, controller=RemoteController, switch=OVSKernelSwitch)
    
    net.start()
    print("*** Network started. Testing connectivity...")
    net.pingAll()
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    run()