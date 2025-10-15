#!/usr/bin/env python3
"""
Port Override for Client Connection
Simple fix to use the actual client port
"""

# Store actual client ports here
CLIENT_PORTS = {
    "client-kabir-1760551531-5e41d5e9": 50847,  # Previous client port
    "client-kabir-1760559400-fb149fa5": 50452,  # Previous client port
    "client-kabir-1760566693-ba541446": 50745,  # Current client port
    # Add more clients as needed
}

def get_client_port(client_id):
    """Get the actual client port"""
    # First check manual overrides
    if client_id in CLIENT_PORTS:
        return CLIENT_PORTS[client_id]
    
    # Try to calculate based on timestamp (more reliable)
    try:
        parts = client_id.split('-')
        if len(parts) >= 3:
            timestamp = int(parts[2])
            port = 50052 + (timestamp % 1000)
            return port
    except:
        pass
    
    # Fallback
    return 50052

def add_client_port(client_id, port):
    """Add a client port dynamically"""
    CLIENT_PORTS[client_id] = port
    print(f"📡 Added client port: {client_id} -> {port}")