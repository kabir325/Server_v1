#!/usr/bin/env python3
"""
Generate gRPC Files from Proto Definition
Creates load_balancer_pb2.py and load_balancer_pb2_grpc.py files
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_grpc_files(proto_file="load_balancer.proto", output_dir="."):
    """Generate gRPC files from proto definition"""
    
    logger.info(f"🔧 Generating gRPC files from {proto_file}")
    logger.info(f"📁 Output directory: {output_dir}")
    
    # Check if proto file exists
    if not os.path.exists(proto_file):
        logger.error(f"❌ Proto file not found: {proto_file}")
        return False
    
    try:
        # Generate Python gRPC files
        cmd = [
            sys.executable, "-m", "grpc_tools.protoc",
            f"--proto_path={os.path.dirname(proto_file) or '.'}",
            f"--python_out={output_dir}",
            f"--grpc_python_out={output_dir}",
            proto_file
        ]
        
        logger.info(f"🚀 Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Check if files were generated
        base_name = os.path.splitext(os.path.basename(proto_file))[0]
        pb2_file = os.path.join(output_dir, f"{base_name}_pb2.py")
        grpc_file = os.path.join(output_dir, f"{base_name}_pb2_grpc.py")
        
        if os.path.exists(pb2_file) and os.path.exists(grpc_file):
            logger.info(f"✅ Generated: {pb2_file}")
            logger.info(f"✅ Generated: {grpc_file}")
            
            # Show file sizes
            pb2_size = os.path.getsize(pb2_file)
            grpc_size = os.path.getsize(grpc_file)
            logger.info(f"📊 File sizes: pb2={pb2_size} bytes, grpc={grpc_size} bytes")
            
            return True
        else:
            logger.error("❌ Generated files not found")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ gRPC generation failed: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False

def install_grpc_tools():
    """Install grpcio-tools if not available"""
    try:
        import grpc_tools
        logger.info("✅ grpcio-tools already installed")
        return True
    except ImportError:
        logger.info("📦 Installing grpcio-tools...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "grpcio-tools"], check=True)
            logger.info("✅ grpcio-tools installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install grpcio-tools: {e}")
            return False

def main():
    """Main function"""
    logger.info("🚀 gRPC File Generator")
    logger.info("=" * 40)
    
    # Install grpcio-tools if needed
    if not install_grpc_tools():
        logger.error("❌ Cannot proceed without grpcio-tools")
        return False
    
    # Generate files for different directories
    directories = [
        ("Server_v1", "Server_v1/load_balancer.proto"),
        ("Client_v1", "Client_v1/load_balancer.proto"),
        ("LB", "LB/load_balancer.proto"),
        (".", "load_balancer.proto")  # Current directory
    ]
    
    success_count = 0
    
    for output_dir, proto_path in directories:
        if os.path.exists(proto_path):
            logger.info(f"\n📁 Processing {proto_path} -> {output_dir}/")
            if generate_grpc_files(proto_path, output_dir):
                success_count += 1
            else:
                logger.warning(f"⚠️ Failed to generate files for {output_dir}")
        else:
            logger.info(f"⏭️ Skipping {proto_path} (not found)")
    
    logger.info(f"\n🎉 Generated gRPC files for {success_count} directories")
    
    if success_count > 0:
        logger.info("\n📋 Generated files:")
        for output_dir, proto_path in directories:
            if os.path.exists(proto_path):
                base_name = os.path.splitext(os.path.basename(proto_path))[0]
                pb2_file = os.path.join(output_dir, f"{base_name}_pb2.py")
                grpc_file = os.path.join(output_dir, f"{base_name}_pb2_grpc.py")
                if os.path.exists(pb2_file):
                    logger.info(f"  ✅ {pb2_file}")
                    logger.info(f"  ✅ {grpc_file}")
    
    return success_count > 0

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)