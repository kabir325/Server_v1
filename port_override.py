#!/usr/bin/env python3
"""
Port Override for Client Connection
Simple fix to use the actual client port
"""

# Store actual client ports here
CLIENT_PORTS = {
    "client-kabir-1760551531-5e41d5e9": 50847,  # Your actual client port
    # Add more clients as needed
}

def get_client_port(client_id):
    """Get the actual client port"""
    return CLIENT_PORTS.get(client_id, 50052)