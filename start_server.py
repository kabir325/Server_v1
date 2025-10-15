#!/usr/bin/env python3
"""
Start AI Load Balancer Server v1.0
Simple startup script for the Raspberry Pi server
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all requirements are installed"""
    try:
        import grpc
        import load_balancer_pb2
        import load_balancer_pb2_grpc
        logger.info("✅ All requirements satisfied")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing requirement: {e}")
        logger.info("💡 Run: pip install -r requirements.txt")
        return False

def generate_grpc_files():
    """Generate gRPC files if they don't exist"""
    if not os.path.exists("load_balancer_pb2.py"):
        logger.info("🔧 Generating gRPC files...")
        try:
            subprocess.run([
                sys.executable, "-m", "grpc_tools.protoc",
                "--proto_path=.",
                "--python_out=.",
                "--grpc_python_out=.",
                "load_balancer.proto"
            ], check=True)
            logger.info("✅ gRPC files generated")
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to generate gRPC files: {e}")
            return False
    return True

def main():
    """Main startup function"""
    logger.info("🚀 Starting AI Load Balancer Server v1.0")
    logger.info("=" * 50)
    
    # Check requirements
    if not check_requirements():
        logger.error("❌ Requirements not met")
        return False
    
    # Generate gRPC files
    if not generate_grpc_files():
        logger.error("❌ gRPC file generation failed")
        return False
    
    # Start the server
    logger.info("🎯 Starting server...")
    try:
        from ai_load_balancer_server import serve
        serve()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)