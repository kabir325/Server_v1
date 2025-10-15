#!/usr/bin/env python3
"""
AI Load Balancer Server v1.0
Raspberry Pi server for distributed LLM processing
"""

import grpc
from concurrent import futures
import threading
import time
import logging
import json
import uuid
import os
import sys
import socket
from typing import Dict, List, Optional
import psutil

# Import generated gRPC files
import load_balancer_pb2
import load_balancer_pb2_grpc

# Import LLM task manager
from llm_task_manager import llm_task_manager, LLM_MODELS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AILoadBalancerServer(load_balancer_pb2_grpc.LoadBalancerServicer):
    """AI Load Balancer Server for distributed LLM processing"""
    
    def __init__(self):
        self.clients: Dict[str, Dict] = {}
        self.processing_requests: Dict[str, Dict] = {}
        self.llm_assignments: Dict[str, str] = {}  # client_id -> model_name
        
        logger.info("🚀 AI Load Balancer Server v1.0 initialized")
        logger.info("🎯 Ready for distributed LLM processing")
    
    def RegisterClient(self, request, context):
        """Register a new client and assign LLM model"""
        try:
            client_id = request.client_id
            # Calculate expected client gRPC port
            expected_port = 50052 + (hash(client_id) % 100)
            
            self.clients[client_id] = {
                "client_info": request,
                "last_heartbeat": time.time(),
                "status": "active",
                "registered_at": time.time(),
                "grpc_port": expected_port  # Store expected gRPC port
            }
            
            logger.info(f"✅ Client registered: {client_id} ({request.hostname})")
            logger.info(f"  CPU: {request.specs.cpu_cores} cores @ {request.specs.cpu_frequency_ghz:.2f} GHz")
            logger.info(f"  RAM: {request.specs.ram_gb} GB")
            logger.info(f"  GPU: {request.specs.gpu_info} ({request.specs.gpu_memory_gb} GB)")
            logger.info(f"  Performance Score: {request.specs.performance_score:.2f}")
            logger.info(f"📊 Total clients: {len(self.clients)}")
            
            # Assign optimal LLM model to this client
            self._assign_llm_models()
            
            return load_balancer_pb2.RegistrationResponse(
                success=True,
                message=f"Client {client_id} registered successfully. Total clients: {len(self.clients)}",
                assigned_id=client_id
            )
            
        except Exception as e:
            logger.error(f"Client registration failed: {e}")
            return load_balancer_pb2.RegistrationResponse(
                success=False,
                message=f"Registration failed: {e}",
                assigned_id=""
            )
    
    def HealthCheck(self, request, context):
        """Health check endpoint"""
        active_clients = len([c for c in self.clients.values() if c['status'] == 'active'])
        return load_balancer_pb2.HealthResponse(
            healthy=True,
            message=f"AI Load Balancer is running with {active_clients} active clients",
            timestamp=int(time.time())
        )
    
    def GetSystemSpecs(self, request, context):
        """Get system status and client information"""
        try:
            # Clean up stale clients (older than 5 minutes)
            current_time = time.time()
            stale_clients = []
            for client_id, client_data in self.clients.items():
                if current_time - client_data['last_heartbeat'] > 300:  # 5 minutes
                    stale_clients.append(client_id)
            
            for client_id in stale_clients:
                logger.info(f"Removing stale client: {client_id}")
                del self.clients[client_id]
            
            # Return system specs
            active_clients = len([c for c in self.clients.values() if c['status'] == 'active'])
            total_models = len(LLM_MODELS)
            deployed_models = len(self.llm_assignments)
            
            return load_balancer_pb2.SystemSpecs(
                cpu_cores=active_clients,
                cpu_frequency_ghz=float(total_models),
                ram_gb=deployed_models,
                gpu_info=f"Load Balancer Status: {active_clients} clients, {total_models} models available, {deployed_models} deployed",
                gpu_memory_gb=0.0,
                os_info="AI Load Balancer Server v1.0",
                performance_score=100.0 if active_clients > 0 else 0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return load_balancer_pb2.SystemSpecs(
                cpu_cores=0,
                cpu_frequency_ghz=0.0,
                ram_gb=0,
                gpu_info=f"Error getting status: {e}",
                gpu_memory_gb=0.0,
                os_info="AI Load Balancer Server v1.0",
                performance_score=0.0
            )
    
    def GetAvailableModels(self, request, context):
        """Get list of available LLM models and their assignments"""
        try:
            models = []
            
            # Add LLM models with their assignments
            for model_name, config in LLM_MODELS.items():
                # Find which client has this model
                assigned_client = None
                for client_id, assigned_model in self.llm_assignments.items():
                    if assigned_model == model_name:
                        assigned_client = client_id
                        break
                
                if assigned_client and assigned_client in self.clients:
                    client_info = self.clients[assigned_client]['client_info']
                    models.append(load_balancer_pb2.ModelInfo(
                        model_name=model_name,
                        model_type="llm",
                        status="available",
                        endpoint_url=f"http://{client_info.ip_address}:{config.gradio_port}",
                        client_id=assigned_client,
                        performance_score=client_info.specs.performance_score
                    ))
                else:
                    # Model not assigned yet
                    models.append(load_balancer_pb2.ModelInfo(
                        model_name=model_name,
                        model_type="llm",
                        status="unassigned",
                        endpoint_url="",
                        client_id="",
                        performance_score=0.0
                    ))
            
            return load_balancer_pb2.AvailableModelsResponse(models=models)
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            return load_balancer_pb2.AvailableModelsResponse(models=[])
    
    def ProcessAIRequest(self, request, context):
        """Process an AI inference request"""
        try:
            logger.info(f"🤖 Processing AI request for model: {request.model_name}")
            logger.info(f"📝 Prompt: {request.prompt[:100]}...")
            
            # Find which client has this model
            assigned_client = None
            for client_id, assigned_model in self.llm_assignments.items():
                if assigned_model == request.model_name:
                    assigned_client = client_id
                    break
            
            if not assigned_client:
                return load_balancer_pb2.AIResponse(
                    request_id=request.request_id,
                    success=False,
                    response_text=f"No client assigned to model {request.model_name}",
                    processing_time=0.0,
                    model_used=request.model_name,
                    client_id=""
                )
            
            # For demo purposes, return a simulated response
            # In production, this would forward to the actual client
            model_config = LLM_MODELS.get(request.model_name)
            if model_config:
                response_text = f"[DEMO] Response from {model_config.model_size} model on client {assigned_client}: This is a simulated agricultural AI response to: '{request.prompt}'. In production, this would be processed by the actual LLM model."
            else:
                response_text = f"[DEMO] Response from {request.model_name}: Simulated response"
            
            return load_balancer_pb2.AIResponse(
                request_id=request.request_id,
                success=True,
                response_text=response_text,
                processing_time=2.5,
                model_used=request.model_name,
                client_id=assigned_client
            )
            
        except Exception as e:
            logger.error(f"AI request processing failed: {e}")
            return load_balancer_pb2.AIResponse(
                request_id=request.request_id,
                success=False,
                response_text=f"Processing failed: {e}",
                processing_time=0.0,
                model_used=request.model_name,
                client_id=""
            )
    
    def _assign_llm_models(self):
        """Assign LLM models to clients based on their capabilities"""
        try:
            logger.info("🔄 Assigning LLM models to clients...")
            
            # Get current assignments
            assignments = llm_task_manager.distribute_models_to_clients(self.clients)
            
            # Update our assignments
            self.llm_assignments = assignments
            
            # Log assignments
            for client_id, model_name in assignments.items():
                if client_id in self.clients:
                    client_info = self.clients[client_id]['client_info']
                    model_config = LLM_MODELS[model_name]
                    logger.info(f"🎯 Assigned {model_name} ({model_config.model_size}) to client {client_id}")
                    logger.info(f"   Client: {client_info.hostname} ({client_info.ip_address})")
                    logger.info(f"   Performance: {client_info.specs.performance_score:.1f}")
            
            logger.info(f"✅ LLM model assignment completed: {len(assignments)} assignments")
            
        except Exception as e:
            logger.error(f"❌ LLM model assignment failed: {e}")
    
    def get_llm_assignments(self) -> Dict[str, str]:
        """Get current LLM model assignments"""
        return self.llm_assignments.copy()
    
    def process_distributed_query(self, prompt: str) -> str:
        """Process a query across all connected clients and summarize results"""
        try:
            logger.info(f"🔄 Processing distributed query: '{prompt[:50]}...'")
            
            # Get active clients
            active_clients = [
                (client_id, client_data) 
                for client_id, client_data in self.clients.items() 
                if client_data['status'] == 'active'
            ]
            
            if not active_clients:
                return "❌ No active clients available for processing"
            
            logger.info(f"📊 Found {len(active_clients)} active clients")
            
            # Send query to all clients
            responses = {}
            for client_id, client_data in active_clients:
                assigned_model = self.llm_assignments.get(client_id)
                if assigned_model:
                    logger.info(f"📤 Sending query to client {client_id} (model: {assigned_model})")
                    
                    # Create AI request
                    request_id = f"req_{int(time.time())}_{client_id}"
                    ai_request = load_balancer_pb2.AIRequest(
                        request_id=request_id,
                        model_name=assigned_model,
                        prompt=prompt,
                        parameters={"temperature": "0.7", "max_tokens": "256"}
                    )
                    
                    # Send request to client via gRPC
                    response = self._send_request_to_client(client_id, ai_request)
                    
                    if response.success:
                        responses[client_id] = {
                            "model": assigned_model,
                            "response": response.response_text,
                            "processing_time": response.processing_time
                        }
                        logger.info(f"✅ Received response from {client_id} ({response.processing_time:.1f}s)")
                    else:
                        logger.warning(f"⚠️ Failed to get response from {client_id}")
            
            # Summarize responses
            if responses:
                summary = self._summarize_responses(prompt, responses)
                logger.info(f"📝 Generated summary from {len(responses)} responses")
                return summary
            else:
                return "❌ No responses received from clients"
                
        except Exception as e:
            logger.error(f"❌ Distributed query processing failed: {e}")
            return f"Error processing query: {str(e)}"
    
    def _summarize_responses(self, original_prompt: str, responses: Dict) -> str:
        """Summarize multiple responses into a single coherent answer"""
        try:
            logger.info("🔄 Summarizing responses from multiple models...")
            
            # Find the best client for summarization (highest performance)
            best_client = None
            best_score = 0
            
            for client_id in responses.keys():
                if client_id in self.clients:
                    score = self.clients[client_id]['client_info'].specs.performance_score
                    if score > best_score:
                        best_score = score
                        best_client = client_id
            
            if not best_client:
                # Fallback: just combine responses
                combined = f"Combined responses for: '{original_prompt}'\n\n"
                for client_id, resp_data in responses.items():
                    model_size = LLM_MODELS[resp_data['model']].model_size
                    combined += f"🤖 {model_size} Model Response:\n{resp_data['response']}\n\n"
                return combined
            
            # Create summarization prompt
            summary_prompt = f"""Please provide a comprehensive summary combining these multiple AI responses to the question: "{original_prompt}"

Responses to summarize:
"""
            
            for client_id, resp_data in responses.items():
                model_size = LLM_MODELS[resp_data['model']].model_size
                summary_prompt += f"\n{model_size} Model Response: {resp_data['response']}\n"
            
            summary_prompt += "\nPlease provide a single, coherent, and comprehensive answer that combines the best insights from all responses:"
            
            logger.info(f"📤 Sending summarization request to best client: {best_client}")
            
            # Send to best client for summarization
            request_id = f"summary_{int(time.time())}"
            assigned_model = self.llm_assignments.get(best_client)
            
            ai_request = load_balancer_pb2.AIRequest(
                request_id=request_id,
                model_name=assigned_model,
                prompt=summary_prompt,
                parameters={"temperature": "0.3", "max_tokens": "512"}  # Lower temp for more focused summary
            )
            
            response = self.ProcessAIRequest(ai_request, None)
            
            if response.success:
                logger.info("✅ Summary generated successfully")
                return f"📋 Comprehensive Summary:\n\n{response.response_text}"
            else:
                # Fallback to simple combination
                logger.warning("⚠️ Summarization failed, using simple combination")
                combined = f"Combined responses for: '{original_prompt}'\n\n"
                for client_id, resp_data in responses.items():
                    model_size = LLM_MODELS[resp_data['model']].model_size
                    combined += f"🤖 {model_size} Model: {resp_data['response']}\n\n"
                return combined
                
        except Exception as e:
            logger.error(f"❌ Summarization failed: {e}")
            # Return simple combination as fallback
            combined = f"Responses for: '{original_prompt}'\n\n"
            for client_id, resp_data in responses.items():
                combined += f"Model {resp_data['model']}: {resp_data['response']}\n\n"
            return combined
    
    def _send_request_to_client(self, client_id: str, ai_request):
        """Send AI request to a specific client"""
        try:
            if client_id not in self.clients:
                logger.error(f"❌ Client {client_id} not found")
                return load_balancer_pb2.AIResponse(
                    request_id=ai_request.request_id,
                    success=False,
                    response_text="Client not found",
                    processing_time=0.0,
                    model_used=ai_request.model_name,
                    client_id=client_id
                )
            
            client_data = self.clients[client_id]
            client_info = client_data['client_info']
            client_ip = client_info.ip_address
            
            # Use stored gRPC port
            client_port = client_data.get('grpc_port', 50052 + (hash(client_id) % 100))
            client_address = f"{client_ip}:{client_port}"
            
            logger.info(f"📡 Connecting to client at {client_address}")
            
            # Test basic connectivity first
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(2)
                result = test_socket.connect_ex((client_ip, client_port))
                test_socket.close()
                
                if result != 0:
                    logger.warning(f"⚠️ Cannot reach client at {client_address} (connection refused)")
                    return load_balancer_pb2.AIResponse(
                        request_id=ai_request.request_id,
                        success=False,
                        response_text=f"Client unreachable at {client_address}",
                        processing_time=0.0,
                        model_used=ai_request.model_name,
                        client_id=client_id
                    )
                else:
                    logger.info(f"✅ Client reachable at {client_address}")
            except Exception as e:
                logger.warning(f"⚠️ Connection test failed: {e}")
            
            # Create gRPC channel to client
            channel = grpc.insecure_channel(client_address)
            stub = load_balancer_pb2_grpc.LoadBalancerStub(channel)
            
            # Send request with timeout
            response = stub.ProcessAIRequest(ai_request, timeout=30)
            
            # Close channel
            channel.close()
            
            return response
            
        except grpc.RpcError as e:
            logger.error(f"❌ gRPC error communicating with client {client_id}: {e}")
            return load_balancer_pb2.AIResponse(
                request_id=ai_request.request_id,
                success=False,
                response_text=f"Communication error: {e.details()}",
                processing_time=0.0,
                model_used=ai_request.model_name,
                client_id=client_id
            )
        except Exception as e:
            logger.error(f"❌ Error sending request to client {client_id}: {e}")
            return load_balancer_pb2.AIResponse(
                request_id=ai_request.request_id,
                success=False,
                response_text=f"Error: {str(e)}",
                processing_time=0.0,
                model_used=ai_request.model_name,
                client_id=client_id
            )
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        active_clients = len([c for c in self.clients.values() if c['status'] == 'active'])
        
        return {
            "total_clients": len(self.clients),
            "active_clients": active_clients,
            "llm_models": len(LLM_MODELS),
            "assigned_models": len(self.llm_assignments),
            "assignments": self.llm_assignments,
            "clients": {
                client_id: {
                    "hostname": client_data['client_info'].hostname,
                    "ip_address": client_data['client_info'].ip_address,
                    "performance_score": client_data['client_info'].specs.performance_score,
                    "status": client_data['status'],
                    "assigned_model": self.llm_assignments.get(client_id, "none")
                }
                for client_id, client_data in self.clients.items()
            }
        }

class InteractiveQueryProcessor:
    """Interactive query processor for real-time distributed LLM queries"""
    
    def __init__(self, server_instance):
        self.server = server_instance
        self.running = False
    
    def start(self):
        """Start the interactive query processor"""
        self.running = True
        logger.info("🎯 Interactive Query Processor started")
        logger.info("💡 Type your queries below (or 'quit' to exit)")
        logger.info("=" * 60)
        
        while self.running:
            try:
                # Get user input
                prompt = input("\n🌾 Enter your agricultural query: ").strip()
                
                if not prompt:
                    continue
                    
                if prompt.lower() in ['quit', 'exit', 'q']:
                    logger.info("👋 Exiting interactive mode...")
                    break
                
                if prompt.lower() == 'status':
                    self._show_system_status()
                    continue
                
                # Process the query
                logger.info("=" * 60)
                logger.info(f"🚀 PROCESSING QUERY: '{prompt}'")
                logger.info("=" * 60)
                
                start_time = time.time()
                result = self.server.process_distributed_query(prompt)
                end_time = time.time()
                
                logger.info("=" * 60)
                logger.info("🎉 FINAL RESULT:")
                logger.info("=" * 60)
                print(f"\n{result}\n")
                logger.info(f"⏱️ Total processing time: {end_time - start_time:.2f} seconds")
                logger.info("=" * 60)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Exiting interactive mode...")
                break
            except EOFError:
                logger.info("\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                logger.error(f"❌ Error in interactive processor: {e}")
    
    def _show_system_status(self):
        """Show current system status"""
        logger.info("📊 SYSTEM STATUS:")
        logger.info("-" * 40)
        
        active_clients = [c for c in self.server.clients.values() if c['status'] == 'active']
        logger.info(f"👥 Active clients: {len(active_clients)}")
        
        for client_id, client_data in self.server.clients.items():
            if client_data['status'] == 'active':
                assigned_model = self.server.llm_assignments.get(client_id, "none")
                model_size = LLM_MODELS.get(assigned_model, {}).get('model_size', 'Unknown')
                performance = client_data['client_info'].specs.performance_score
                
                logger.info(f"  🖥️ {client_id[:20]}... -> {model_size} model (score: {performance:.1f})")
        
        logger.info("-" * 40)

def serve():
    """Start the AI Load Balancer server"""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    
    # Add the service
    ai_service = AILoadBalancerServer()
    load_balancer_pb2_grpc.add_LoadBalancerServicer_to_server(ai_service, server)
    
    # Listen on all interfaces
    listen_addr = '0.0.0.0:50051'
    server.add_insecure_port(listen_addr)
    
    # Start server
    server.start()
    
    logger.info("🚀 AI Load Balancer Server v1.0 started")
    logger.info(f"🌐 Listening on {listen_addr}")
    logger.info("🎯 Ready for distributed LLM processing")
    logger.info("📊 Supported models:")
    for model_name, config in LLM_MODELS.items():
        logger.info(f"   - {model_name} ({config.model_size})")
    
    # Start interactive query processor
    query_processor = InteractiveQueryProcessor(ai_service)
    query_thread = threading.Thread(target=query_processor.start, daemon=True)
    query_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down server...")
        server.stop(0)

if __name__ == '__main__':
    serve()