#!/usr/bin/env python3
"""
Test Distributed Query Processing
Simple script to test the distributed LLM system
"""

import sys
import time
import logging

# Add current directory to path for imports
sys.path.append('.')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_distributed_query():
    """Test the distributed query system"""
    try:
        # Import after adding path
        from ai_load_balancer_server import AILoadBalancerServer
        
        logger.info("🧪 Testing Distributed Query System")
        logger.info("=" * 50)
        
        # Create server instance
        server = AILoadBalancerServer()
        
        # Simulate some clients (for testing)
        import load_balancer_pb2
        
        # Mock client 1 (high performance)
        client1_specs = load_balancer_pb2.SystemSpecs(
            cpu_cores=8,
            cpu_frequency_ghz=3.2,
            ram_gb=16,
            gpu_info="NVIDIA RTX 3070",
            gpu_memory_gb=8.0,
            os_info="Windows 11",
            performance_score=85.0
        )
        
        client1_info = load_balancer_pb2.ClientInfo(
            client_id="test-client-1",
            hostname="laptop1",
            ip_address="192.168.1.100",
            specs=client1_specs
        )
        
        # Mock client 2 (medium performance)
        client2_specs = load_balancer_pb2.SystemSpecs(
            cpu_cores=6,
            cpu_frequency_ghz=2.8,
            ram_gb=12,
            gpu_info="NVIDIA GTX 1660",
            gpu_memory_gb=6.0,
            os_info="Windows 10",
            performance_score=65.0
        )
        
        client2_info = load_balancer_pb2.ClientInfo(
            client_id="test-client-2",
            hostname="laptop2",
            ip_address="192.168.1.101",
            specs=client2_specs
        )
        
        # Register mock clients
        server.clients["test-client-1"] = {
            "client_info": client1_info,
            "last_heartbeat": time.time(),
            "status": "active",
            "registered_at": time.time()
        }
        
        server.clients["test-client-2"] = {
            "client_info": client2_info,
            "last_heartbeat": time.time(),
            "status": "active",
            "registered_at": time.time()
        }
        
        # Assign models
        server._assign_llm_models()
        
        # Test query
        test_prompt = "What are the best practices for organic farming in India?"
        
        logger.info(f"🚀 Testing query: '{test_prompt}'")
        result = server.process_distributed_query(test_prompt)
        
        logger.info("🎉 Test Result:")
        logger.info("=" * 50)
        print(result)
        logger.info("=" * 50)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_distributed_query()
    if not success:
        sys.exit(1)