"""
Ryu SDN Controller integrating the Dueling DQN Load Balancer.
Run this alongside Mininet to route traffic intelligently.
"""
from ryu.lib import hub
from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4
import numpy as np

# Import our RL Selector
from select_server import Selector, build_state

class DQNLoadBalancer(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DQNLoadBalancer, self).__init__(*args, **kwargs)
        # Load the trained model
        self.selector = Selector("results/models/dueling_seed0.pt")
        self.mac_to_port = {}
        self.server_ports = [1, 2, 3] # Assuming servers are on ports 1, 2, 3
        

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority, match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        
        # If packet is from a client (not a server), route it using DQN
        if in_port not in self.server_ports:
            # Build the state vector
            drain_times = [self.server_metrics[p] for p in self.server_ports]
            conns = [10, 10, 10] # Placeholder; in reality, parse active flows
            lats = [0.1, 0.1, 0.1]  # Placeholder; calculate from stats
            req_size = 1.0 # Placeholder; can be parsed from IP packet length
            
            state = build_state(drain_times, conns, lats, req_size)
            
            # Query the DQN agent
            out_port = self.server_ports[self.selector.select_server(state)]
            
            # Install flow rule
            match = parser.OFPMatch(in_port=in_port, eth_dst=eth.dst)
            actions = [parser.OFPActionOutput(out_port)]
            self.add_flow(datapath, 10, match, actions)
            
            # Send packet out
            out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, 
                                      in_port=in_port, actions=actions, data=msg.data)
            datapath.send_msg(out)
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        
        # Learn MAC address
        self.mac_to_port.setdefault(datapath.id, {})
        self.mac_to_port[datapath.id][eth.src] = in_port
        
        # If we know the destination port, send it directly. 
        # Otherwise, flood it (which solves the ARP problem for pings).
        if eth.dst in self.mac_to_port[datapath.id]:
            out_port = self.mac_to_port[datapath.id][eth.dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        actions = [parser.OFPActionOutput(out_port)]
        
        # --- DQN LOGIC ---
        # If a client (Port 4) is sending data to a server, use the DQN!
        if in_port == 4 and out_port != ofproto.OFPP_FLOOD:
            drain_times = [0.1, 0.1, 0.1]
            conns = [10, 10, 10] 
            lats = [0.1, 0.1, 0.1]  
            req_size = 1.0 
            
            state = build_state(drain_times, conns, lats, req_size)
            chosen_index = self.selector.select_server(state)
            out_port = self.server_ports[chosen_index]
            actions = [parser.OFPActionOutput(out_port)]
            
            self.logger.info(f"FLOW INTERCEPTED! DQN chose server index {chosen_index} (Port {out_port})")

            match = parser.OFPMatch(in_port=in_port, eth_dst=eth.dst)
            self.add_flow(datapath, 10, match, actions)

        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id, 
                                  in_port=in_port, actions=actions, data=msg.data)
        datapath.send_msg(out)
