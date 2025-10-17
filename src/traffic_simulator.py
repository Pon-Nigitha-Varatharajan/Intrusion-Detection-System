"""
Network Traffic Simulator for Testing Real-Time Predictions
Generates synthetic network flows based on typical patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class NetworkTrafficSimulator:
    """Generate synthetic network traffic for testing"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.feature_ranges = self._define_feature_ranges()
    
    def _define_feature_ranges(self):
        """Define typical ranges for network features"""
        return {
            # Normal traffic patterns
            'normal': {
                'flow_duration': (0, 120000000),  # 0-120 seconds in microseconds
                'total_fwd_packets': (1, 50),
                'total_bwd_packets': (1, 50),
                'flow_bytes_s': (100, 10000),
                'flow_packets_s': (1, 100),
                'fwd_packet_length_mean': (40, 1500),
                'bwd_packet_length_mean': (40, 1500),
                'flow_iat_mean': (1000, 1000000),
                'fwd_iat_mean': (1000, 1000000),
                'bwd_iat_mean': (1000, 1000000),
                'active_mean': (10000, 5000000),
                'idle_mean': (10000, 10000000),
            },
            # Attack patterns (more aggressive)
            'attack': {
                'flow_duration': (0, 10000000),  # Shorter duration
                'total_fwd_packets': (50, 1000),  # More packets
                'total_bwd_packets': (0, 10),  # Few responses
                'flow_bytes_s': (50000, 1000000),  # High throughput
                'flow_packets_s': (100, 10000),  # High packet rate
                'fwd_packet_length_mean': (40, 200),  # Smaller packets
                'bwd_packet_length_mean': (0, 100),
                'flow_iat_mean': (0, 10000),  # Low inter-arrival time
                'fwd_iat_mean': (0, 10000),
                'bwd_iat_mean': (0, 100000),
                'active_mean': (1000, 100000),
                'idle_mean': (0, 100000),
            }
        }
    
    def generate_flow(self, flow_type='normal'):
        """Generate a single network flow"""
        ranges = self.feature_ranges[flow_type]
        
        flow = {}
        for feature, (min_val, max_val) in ranges.items():
            if flow_type == 'attack' and feature in ['total_fwd_packets', 'flow_bytes_s', 'flow_packets_s']:
                # Skewed distribution for attack traffic
                flow[feature] = np.random.exponential(scale=(max_val - min_val) / 3) + min_val
                flow[feature] = min(flow[feature], max_val)
            else:
                # Normal distribution for regular features
                mean = (min_val + max_val) / 2
                std = (max_val - min_val) / 6
                flow[feature] = np.random.normal(mean, std)
                flow[feature] = np.clip(flow[feature], min_val, max_val)
        
        # Ensure minimum values for packet counts
        flow['total_fwd_packets'] = max(1, flow['total_fwd_packets'])
        flow['total_bwd_packets'] = max(1, flow['total_bwd_packets'])
        
        # Add derived features
        flow['fwd_header_length'] = flow['total_fwd_packets'] * np.random.uniform(20, 60)
        flow['bwd_header_length'] = flow['total_bwd_packets'] * np.random.uniform(20, 60)
        
        # Protocol (TCP=6, UDP=17)
        flow['protocol'] = 6 if np.random.random() > 0.3 else 17
        
        # Flags - ensure valid ranges
        flow['fwd_psh_flags'] = np.random.randint(0, max(2, int(flow['total_fwd_packets'] / 2) + 1))
        flow['bwd_psh_flags'] = np.random.randint(0, max(2, int(flow['total_bwd_packets'] / 2) + 1))
        flow['fwd_urg_flags'] = np.random.randint(0, 3)
        flow['bwd_urg_flags'] = np.random.randint(0, 3)
        
        # Additional common network features
        flow['fwd_packet_length_max'] = flow['fwd_packet_length_mean'] * np.random.uniform(1.2, 2.0)
        flow['fwd_packet_length_min'] = flow['fwd_packet_length_mean'] * np.random.uniform(0.1, 0.8)
        flow['fwd_packet_length_std'] = (flow['fwd_packet_length_max'] - flow['fwd_packet_length_min']) / 4
        
        flow['bwd_packet_length_max'] = flow['bwd_packet_length_mean'] * np.random.uniform(1.2, 2.0)
        flow['bwd_packet_length_min'] = flow['bwd_packet_length_mean'] * np.random.uniform(0.1, 0.8)
        flow['bwd_packet_length_std'] = (flow['bwd_packet_length_max'] - flow['bwd_packet_length_min']) / 4
        
        # Inter-arrival time stats
        flow['fwd_iat_max'] = flow['fwd_iat_mean'] * np.random.uniform(2, 5)
        flow['fwd_iat_min'] = flow['fwd_iat_mean'] * np.random.uniform(0.1, 0.5)
        flow['fwd_iat_std'] = (flow['fwd_iat_max'] - flow['fwd_iat_min']) / 3
        
        flow['bwd_iat_max'] = flow['bwd_iat_mean'] * np.random.uniform(2, 5)
        flow['bwd_iat_min'] = flow['bwd_iat_mean'] * np.random.uniform(0.1, 0.5)
        flow['bwd_iat_std'] = (flow['bwd_iat_max'] - flow['bwd_iat_min']) / 3
        
        flow['flow_iat_max'] = max(flow['fwd_iat_max'], flow['bwd_iat_max'])
        flow['flow_iat_min'] = min(flow['fwd_iat_min'], flow['bwd_iat_min'])
        flow['flow_iat_std'] = (flow['flow_iat_max'] - flow['flow_iat_min']) / 3
        
        # Bulk statistics
        flow['fwd_bulk_bytes'] = flow['total_fwd_packets'] * flow['fwd_packet_length_mean'] * np.random.uniform(0.3, 0.8)
        flow['fwd_bulk_packets'] = int(flow['total_fwd_packets'] * np.random.uniform(0.3, 0.8))
        flow['fwd_bulk_rate'] = flow['fwd_bulk_bytes'] / max(1, flow['flow_duration'])
        
        flow['bwd_bulk_bytes'] = flow['total_bwd_packets'] * flow['bwd_packet_length_mean'] * np.random.uniform(0.3, 0.8)
        flow['bwd_bulk_packets'] = int(flow['total_bwd_packets'] * np.random.uniform(0.3, 0.8))
        flow['bwd_bulk_rate'] = flow['bwd_bulk_bytes'] / max(1, flow['flow_duration'])
        
        # Subflow statistics
        flow['fwd_subflow_packets'] = flow['total_fwd_packets']
        flow['bwd_subflow_packets'] = flow['total_bwd_packets']
        flow['fwd_subflow_bytes'] = flow['total_fwd_packets'] * flow['fwd_packet_length_mean']
        flow['bwd_subflow_bytes'] = flow['total_bwd_packets'] * flow['bwd_packet_length_mean']
        
        # Window sizes
        flow['fwd_init_win_bytes'] = np.random.randint(0, 65535)
        flow['bwd_init_win_bytes'] = np.random.randint(0, 65535)
        
        # Segment sizes
        flow['fwd_avg_segment_size'] = flow['fwd_packet_length_mean']
        flow['bwd_avg_segment_size'] = flow['bwd_packet_length_mean']
        
        # Add label
        flow['Label'] = 1 if flow_type == 'attack' else 0
        
        return flow
    
    def generate_batch(self, n_samples=100, attack_ratio=0.3):
        """Generate a batch of network flows"""
        flows = []
        
        n_attacks = int(n_samples * attack_ratio)
        n_normal = n_samples - n_attacks
        
        # Generate normal traffic
        for _ in range(n_normal):
            flows.append(self.generate_flow('normal'))
        
        # Generate attack traffic
        for _ in range(n_attacks):
            flows.append(self.generate_flow('attack'))
        
        # Shuffle
        df = pd.DataFrame(flows)
        df = df.sample(frac=1).reset_index(drop=True)
        
        return df
    
    def generate_streaming_data(self, duration_seconds=60, samples_per_second=1, attack_probability=0.2):
        """Generate streaming data over time"""
        total_samples = duration_seconds * samples_per_second
        flows = []
        
        start_time = datetime.now()
        
        for i in range(total_samples):
            # Determine if this flow is an attack
            is_attack = np.random.random() < attack_probability
            flow_type = 'attack' if is_attack else 'normal'
            
            # Generate flow
            flow = self.generate_flow(flow_type)
            
            # Add timestamp
            flow['timestamp'] = (start_time + timedelta(seconds=i/samples_per_second)).isoformat()
            
            flows.append(flow)
        
        return pd.DataFrame(flows)
    
    def generate_attack_scenario(self, scenario_type='ddos', n_samples=100):
        """Generate specific attack scenarios"""
        flows = []
        
        if scenario_type == 'ddos':
            # DDoS: High packet rate, short duration, many connections
            for _ in range(n_samples):
                flow = self.generate_flow('attack')
                flow['flow_packets_s'] = np.random.uniform(5000, 10000)
                flow['total_fwd_packets'] = np.random.uniform(500, 1000)
                flow['flow_duration'] = np.random.uniform(1000, 5000000)
                flows.append(flow)
        
        elif scenario_type == 'port_scan':
            # Port Scan: Many short connections, varied ports
            for _ in range(n_samples):
                flow = self.generate_flow('attack')
                flow['flow_duration'] = np.random.uniform(100, 1000000)
                flow['total_fwd_packets'] = np.random.uniform(1, 5)
                flow['total_bwd_packets'] = np.random.uniform(0, 2)
                flows.append(flow)
        
        elif scenario_type == 'brute_force':
            # Brute Force: Repeated similar patterns
            for _ in range(n_samples):
                flow = self.generate_flow('attack')
                flow['fwd_packet_length_mean'] = np.random.uniform(100, 300)
                flow['flow_iat_mean'] = np.random.uniform(1000, 10000)
                flows.append(flow)
        
        else:  # mixed attacks
            third = n_samples // 3
            flows.extend(self.generate_attack_scenario('ddos', third)['flows'])
            flows.extend(self.generate_attack_scenario('port_scan', third)['flows'])
            flows.extend(self.generate_attack_scenario('brute_force', n_samples - 2*third)['flows'])
        
        df = pd.DataFrame(flows)
        return df
    
    def generate_test_csv(self, filepath='data/test/simulated_traffic.csv', n_samples=1000, attack_ratio=0.3):
        """Generate and save test CSV file"""
        df = self.generate_batch(n_samples, attack_ratio)
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        df.to_csv(filepath, index=False)
        print(f"✅ Generated {n_samples} flows and saved to {filepath}")
        print(f"   - Normal: {(df['Label'] == 0).sum()}")
        print(f"   - Attacks: {(df['Label'] == 1).sum()}")
        
        return df

# Example usage
if __name__ == "__main__":
    simulator = NetworkTrafficSimulator()
    
    # Generate test data
    df = simulator.generate_test_csv(n_samples=500, attack_ratio=0.3)
    print("\nSample flows:")
    print(df.head())
    
    # Generate streaming data
    streaming_df = simulator.generate_streaming_data(duration_seconds=30, samples_per_second=2)
    streaming_df.to_csv('data/test/streaming_traffic.csv', index=False)
    print("\n✅ Generated streaming traffic data")
    
    # Generate attack scenarios
    ddos_df = simulator.generate_attack_scenario('ddos', n_samples=100)
    ddos_df.to_csv('data/test/ddos_scenario.csv', index=False)
    print("✅ Generated DDoS scenario")