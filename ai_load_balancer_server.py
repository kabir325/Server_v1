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

# Import port override
try:
    from port_override import get_client_port
    USE_PORT_OVERRIDE = True
except ImportError:
    USE_PORT_OVERRIDE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AILoadBalancerServer(load_balancer_pb2_grpc.LoadBalancerServicer):
    """AI Load Balancer Server for distributed LLM processing"""
    
    def __init__(self):
        self.clients: Dict[str, Dict] = {}
        self.processing_requests: Dict[str, Dict] = {}
        self.llm_assignments: Dict[str, str] = {}  # client_id -> model_name
        
        logger.info("AI Load Balancer Server v1.0 initialized")
        logger.info("Ready for distributed LLM processing")
    
    def RegisterClient(self, request, context):
        """Register a new client and assign LLM model"""
        try:
            client_id = request.client_id
            # Calculate expected client gRPC port using same logic as client
            try:
                parts = client_id.split('-')
                if len(parts) >= 3:
                    timestamp = int(parts[2])
                    expected_port = 50052 + (timestamp % 1000)
                else:
                    expected_port = 50052 + (abs(hash(client_id)) % 1000)
            except:
                expected_port = 50052 + (abs(hash(client_id)) % 1000)
            
            self.clients[client_id] = {
                "client_info": request,
                "last_heartbeat": time.time(),
                "status": "active",
                "registered_at": time.time(),
                "grpc_port": expected_port  # Store expected gRPC port
            }
            
            logger.info(f"Client registered: {client_id} ({request.hostname})")
            logger.info(f"  CPU: {request.specs.cpu_cores} cores @ {request.specs.cpu_frequency_ghz:.2f} GHz")
            logger.info(f"  RAM: {request.specs.ram_gb} GB")
            logger.info(f"  GPU: {request.specs.gpu_info} ({request.specs.gpu_memory_gb} GB)")
            logger.info(f"  Performance Score: {request.specs.performance_score:.2f}")
            logger.info(f"Total clients: {len(self.clients)}")
            
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
            logger.info(f"Processing AI request for model: {request.model_name}")
            logger.info(f"Prompt: {request.prompt[:100]}...")
            
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
            logger.info("Assigning LLM models to clients...")
            
            # Get current assignments
            assignments = llm_task_manager.distribute_models_to_clients(self.clients)
            
            # Update our assignments
            self.llm_assignments = assignments
            
            # Log assignments
            for client_id, model_name in assignments.items():
                if client_id in self.clients:
                    client_info = self.clients[client_id]['client_info']
                    model_config = LLM_MODELS[model_name]
                    logger.info(f"Assigned {model_name} ({model_config.model_size}) to client {client_id}")
                    logger.info(f"   Client: {client_info.hostname} ({client_info.ip_address})")
                    logger.info(f"   Performance: {client_info.specs.performance_score:.1f}")
            
            logger.info(f"LLM model assignment completed: {len(assignments)} assignments")
            
        except Exception as e:
            logger.error(f"LLM model assignment failed: {e}")
    
    def get_llm_assignments(self) -> Dict[str, str]:
        """Get current LLM model assignments"""
        return self.llm_assignments.copy()
    
    def process_distributed_query(self, prompt: str) -> str:
        """Process a query across all connected clients and summarize results - DETAILED DATA FLOW"""
        try:
            logger.info(f"[SERVER DATA FLOW] ==========================================")
            logger.info(f"[SERVER DATA FLOW] STARTING DISTRIBUTED QUERY PROCESSING")
            logger.info(f"[SERVER DATA FLOW] Query: '{prompt}'")
            logger.info(f"[SERVER DATA FLOW] ==========================================")
            
            # Get active clients with detailed info
            active_clients = [
                (client_id, client_data) 
                for client_id, client_data in self.clients.items() 
                if client_data['status'] == 'active'
            ]
            
            if not active_clients:
                error_msg = "CRITICAL: No active clients available for processing"
                logger.error(f"[SERVER DATA FLOW] {error_msg}")
                raise Exception(error_msg)
            
            logger.info(f"[SERVER DATA FLOW] Found {len(active_clients)} active clients")
            
            # Show model distribution
            logger.info(f"[SERVER DATA FLOW] MODEL DISTRIBUTION:")
            for client_id, client_data in active_clients:
                assigned_model = self.llm_assignments.get(client_id)
                client_info = client_data['client_info']
                performance = client_info.specs.performance_score
                logger.info(f"[SERVER DATA FLOW]   Client: {client_id[:20]}...")
                logger.info(f"[SERVER DATA FLOW]   Performance: {performance:.2f}")
                logger.info(f"[SERVER DATA FLOW]   Assigned Model: {assigned_model}")
                logger.info(f"[SERVER DATA FLOW]   Hostname: {client_info.hostname}")
                logger.info(f"[SERVER DATA FLOW]   IP: {client_info.ip_address}")
            
            # Send query to all clients
            responses = {}
            logger.info(f"[SERVER DATA FLOW] ==========================================")
            logger.info(f"[SERVER DATA FLOW] SENDING QUERIES TO CLIENTS")
            logger.info(f"[SERVER DATA FLOW] ==========================================")
            
            for client_id, client_data in active_clients:
                assigned_model = self.llm_assignments.get(client_id)
                if assigned_model:
                    logger.info(f"[SERVER DATA FLOW] Sending to client {client_id}")
                    logger.info(f"[SERVER DATA FLOW] Model: {assigned_model}")
                    
                    # Create AI request
                    request_id = f"req_{int(time.time())}_{client_id}"
                    ai_request = load_balancer_pb2.AIRequest(
                        request_id=request_id,
                        model_name=assigned_model,
                        prompt=prompt,
                        parameters={"temperature": "0.7", "max_tokens": "256"}
                    )
                    
                    logger.info(f"[SERVER DATA FLOW] Request ID: {request_id}")
                    
                    # Send request to client via gRPC
                    response = self._send_request_to_client(client_id, ai_request)
                    
                    if response.success:
                        responses[client_id] = {
                            "model": assigned_model,
                            "response": response.response_text,
                            "processing_time": response.processing_time,
                            "client_id": client_id
                        }
                        logger.info(f"[SERVER DATA FLOW] SUCCESS: Received from {client_id}")
                        logger.info(f"[SERVER DATA FLOW] Processing time: {response.processing_time:.1f}s")
                        logger.info(f"[SERVER DATA FLOW] Response preview: '{response.response_text[:100]}...'")
                    else:
                        error_msg = f"FAILED to get response from {client_id}: {response.response_text}"
                        logger.error(f"[SERVER DATA FLOW] {error_msg}")
                        # Don't continue with failed responses - we want to see actual issues
                        raise Exception(error_msg)
            
            # Summarize responses
            if responses:
                logger.info(f"[SERVER DATA FLOW] ==========================================")
                logger.info(f"[SERVER DATA FLOW] STARTING RESPONSE SUMMARIZATION")
                logger.info(f"[SERVER DATA FLOW] Total responses received: {len(responses)}")
                logger.info(f"[SERVER DATA FLOW] ==========================================")
                
                summary = self._summarize_responses(prompt, responses)
                
                logger.info(f"[SERVER DATA FLOW] ==========================================")
                logger.info(f"[SERVER DATA FLOW] SUMMARIZATION COMPLETED")
                logger.info(f"[SERVER DATA FLOW] Final summary length: {len(summary)} chars")
                logger.info(f"[SERVER DATA FLOW] ==========================================")
                
                return summary
            else:
                error_msg = "CRITICAL: No responses received from any clients"
                logger.error(f"[SERVER DATA FLOW] {error_msg}")
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"[SERVER DATA FLOW] CRITICAL ERROR: Distributed query processing failed: {e}")
            raise Exception(f"Distributed processing failed: {e}")
    
    def _summarize_responses(self, original_prompt: str, responses: Dict) -> str:
        """Summarize multiple responses into a single coherent answer - DETAILED DATA FLOW"""
        try:
            logger.info(f"[SUMMARIZATION DATA FLOW] ==========================================")
            logger.info(f"[SUMMARIZATION DATA FLOW] STARTING RESPONSE SUMMARIZATION")
            logger.info(f"[SUMMARIZATION DATA FLOW] Original prompt: '{original_prompt}'")
            logger.info(f"[SUMMARIZATION DATA FLOW] Number of responses: {len(responses)}")
            logger.info(f"[SUMMARIZATION DATA FLOW] ==========================================")
            
            # Show all responses received
            for client_id, resp_data in responses.items():
                model_size = LLM_MODELS[resp_data['model']].model_size
                logger.info(f"[SUMMARIZATION DATA FLOW] Response from {client_id}:")
                logger.info(f"[SUMMARIZATION DATA FLOW]   Model: {resp_data['model']} ({model_size})")
                logger.info(f"[SUMMARIZATION DATA FLOW]   Processing time: {resp_data['processing_time']:.1f}s")
                logger.info(f"[SUMMARIZATION DATA FLOW]   Response: '{resp_data['response'][:150]}...'")
            
            # Find the best client for summarization (highest performance)
            best_client = None
            best_score = 0
            
            logger.info(f"[SUMMARIZATION DATA FLOW] Finding best client for summarization...")
            
            for client_id in responses.keys():
                if client_id in self.clients:
                    score = self.clients[client_id]['client_info'].specs.performance_score
                    logger.info(f"[SUMMARIZATION DATA FLOW]   {client_id}: performance {score:.2f}")
                    if score > best_score:
                        best_score = score
                        best_client = client_id
            
            if not best_client:
                error_msg = "CRITICAL: No suitable client found for summarization"
                logger.error(f"[SUMMARIZATION DATA FLOW] {error_msg}")
                raise Exception(error_msg)
            
            logger.info(f"[SUMMARIZATION DATA FLOW] Selected best client: {best_client} (performance: {best_score:.2f})")
            
            # If only one response, return it directly with clear labeling
            if len(responses) == 1:
                client_id = list(responses.keys())[0]
                resp_data = responses[client_id]
                model_size = LLM_MODELS[resp_data['model']].model_size
                logger.info(f"[SUMMARIZATION DATA FLOW] Only one response - returning directly")
                
                result = f"SINGLE MODEL RESPONSE:\n"
                result += f"Model: {resp_data['model']} ({model_size})\n"
                result += f"Client: {client_id}\n"
                result += f"Processing Time: {resp_data['processing_time']:.1f}s\n"
                result += f"Response: {resp_data['response']}"
                return result
            
            # Create summarization prompt
            logger.info(f"[SUMMARIZATION DATA FLOW] Creating summarization prompt...")
            
            summary_prompt = f"""Please provide a comprehensive summary combining these multiple AI responses to the question: "{original_prompt}"

Responses to summarize:
"""
            
            for client_id, resp_data in responses.items():
                model_size = LLM_MODELS[resp_data['model']].model_size
                summary_prompt += f"\n{model_size} Model Response: {resp_data['response']}\n"
            
            summary_prompt += "\nPlease provide a single, coherent, and comprehensive answer that combines the best insights from all responses:"
            
            logger.info(f"[SUMMARIZATION DATA FLOW] Summarization prompt length: {len(summary_prompt)} chars")
            logger.info(f"[SUMMARIZATION DATA FLOW] Sending to best client: {best_client}")
            
            # Send to best client for summarization
            request_id = f"summary_{int(time.time())}"
            assigned_model = self.llm_assignments.get(best_client)
            
            ai_request = load_balancer_pb2.AIRequest(
                request_id=request_id,
                model_name=assigned_model,
                prompt=summary_prompt,
                parameters={"temperature": "0.3", "max_tokens": "512"}  # Lower temp for more focused summary
            )
            
            logger.info(f"[SUMMARIZATION DATA FLOW] Sending summarization request...")
            response = self.ProcessAIRequest(ai_request, None)
            
            if response.success:
                logger.info(f"[SUMMARIZATION DATA FLOW] SUCCESS: Summary generated by {best_client}")
                logger.info(f"[SUMMARIZATION DATA FLOW] Summary length: {len(response.response_text)} chars")
                logger.info(f"[SUMMARIZATION DATA FLOW] Summary preview: '{response.response_text[:150]}...'")
                
                result = f"MULTI-MODEL SUMMARIZED RESPONSE:\n"
                result += f"Summarized by: {assigned_model} on {best_client}\n"
                result += f"Original responses from {len(responses)} models\n"
                result += f"Summary: {response.response_text}"
                return result
            else:
                # NO FALLBACKS - Force error to show the issue
                error_msg = f"CRITICAL: Summarization failed on {best_client}: {response.response_text}"
                logger.error(f"[SUMMARIZATION DATA FLOW] {error_msg}")
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"[SUMMARIZATION DATA FLOW] CRITICAL ERROR: Summarization failed: {e}")
            raise Exception(f"Summarization failed: {e}")
    
    def _send_request_to_client(self, client_id: str, ai_request):
        """Send AI request to a specific client"""
        try:
            if client_id not in self.clients:
                logger.error(f"Client {client_id} not found")
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
            
            # Use port override if available, otherwise calculate
            if USE_PORT_OVERRIDE:
                client_port = get_client_port(client_id)
                logger.info(f"Using port override: {client_port}")
                
                # If port override fails, try to find the actual port
                if client_port == 50052:  # Default fallback port
                    logger.info(f"Port override not found, trying to detect actual port...")
                    detected_port = self._detect_client_port(client_ip, client_id)
                    if detected_port:
                        client_port = detected_port
                        # Store it for future use
                        from port_override import add_client_port
                        add_client_port(client_id, client_port)
            else:
                # Calculate port using same deterministic logic as client
                try:
                    parts = client_id.split('-')
                    if len(parts) >= 3:
                        timestamp = int(parts[2])
                        client_port = 50052 + (timestamp % 1000)  # Use last 3 digits of timestamp
                    else:
                        client_port = 50052 + (abs(hash(client_id)) % 1000)
                except:
                    client_port = 50052 + (abs(hash(client_id)) % 1000)
            client_address = f"{client_ip}:{client_port}"
            
            logger.info(f"Connecting to client at {client_address}")
            
            # Test basic connectivity first
            try:
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.settimeout(2)
                result = test_socket.connect_ex((client_ip, client_port))
                test_socket.close()
                
                if result != 0:
                    logger.warning(f"Cannot reach client at {client_address} (connection refused)")
                    return load_balancer_pb2.AIResponse(
                        request_id=ai_request.request_id,
                        success=False,
                        response_text=f"Client unreachable at {client_address}",
                        processing_time=0.0,
                        model_used=ai_request.model_name,
                        client_id=client_id
                    )
                else:
                    logger.info(f"Client reachable at {client_address}")
            except Exception as e:
                logger.warning(f"Connection test failed: {e}")
            
            # Create gRPC channel to client
            channel = grpc.insecure_channel(client_address)
            stub = load_balancer_pb2_grpc.LoadBalancerStub(channel)
            
            # Send request with timeout
            response = stub.ProcessAIRequest(ai_request, timeout=30)
            
            # Close channel
            channel.close()
            
            return response
            
        except grpc.RpcError as e:
            logger.error(f"gRPC error communicating with client {client_id}: {e}")
            return load_balancer_pb2.AIResponse(
                request_id=ai_request.request_id,
                success=False,
                response_text=f"Communication error: {e.details()}",
                processing_time=0.0,
                model_used=ai_request.model_name,
                client_id=client_id
            )
        except Exception as e:
            logger.error(f"Error sending request to client {client_id}: {e}")
            return load_balancer_pb2.AIResponse(
                request_id=ai_request.request_id,
                success=False,
                response_text=f"Error: {str(e)}",
                processing_time=0.0,
                model_used=ai_request.model_name,
                client_id=client_id
            )
    
    def _detect_client_port(self, client_ip: str, client_id: str) -> Optional[int]:
        """Try to detect the actual client gRPC port by scanning"""
        try:
            logger.info(f"Scanning for client gRPC port on {client_ip}...")
            
            # Try common port ranges (quick scan)
            common_ports = [50052, 50053, 50054, 50055, 50745, 50452, 50847]  # Include known ports
            
            for port in common_ports:
                try:
                    test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    test_socket.settimeout(0.5)  # Quick test
                    result = test_socket.connect_ex((client_ip, port))
                    test_socket.close()
                    
                    if result == 0:  # Connection successful
                        logger.info(f"Found open port: {port}")
                        return port
                        
                except:
                    continue
            
            logger.warning(f"Could not detect client gRPC port on {client_ip}")
            return None
            
        except Exception as e:
            logger.error(f"Port detection failed: {e}")
            return None
    
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
        logger.info("Interactive Query Processor started")
        logger.info("Type your queries below (or 'quit' to exit)")
        logger.info("=" * 60)
        
        while self.running:
            try:
                # Get user input
                prompt = input("\nEnter your agricultural query: ").strip()
                
                if not prompt:
                    continue
                    
                if prompt.lower() in ['quit', 'exit', 'q']:
                    logger.info("👋 Exiting interactive mode...")
                    break
                
                if prompt.lower() == 'status':
                    self._show_system_status()
                    continue
                
                # Process the query
                logger.info("=" * 80)
                logger.info(f"[MAIN DATA FLOW] PROCESSING QUERY: '{prompt}'")
                logger.info("=" * 80)
                
                start_time = time.time()
                
                try:
                    result = self.server.process_distributed_query(prompt)
                    end_time = time.time()
                    
                    logger.info("=" * 80)
                    logger.info("[MAIN DATA FLOW] FINAL RESULT:")
                    logger.info("=" * 80)
                    print(f"\n{result}\n")
                    logger.info(f"[MAIN DATA FLOW] Total processing time: {end_time - start_time:.2f} seconds")
                    logger.info("=" * 80)
                    
                except Exception as e:
                    end_time = time.time()
                    logger.error("=" * 80)
                    logger.error(f"[MAIN DATA FLOW] PROCESSING FAILED: {e}")
                    logger.error("=" * 80)
                    print(f"\nERROR: {e}\n")
                    logger.error(f"[MAIN DATA FLOW] Failed after: {end_time - start_time:.2f} seconds")
                    logger.error("=" * 80)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Exiting interactive mode...")
                break
            except EOFError:
                logger.info("\n👋 Exiting interactive mode...")
                break
            except Exception as e:
                logger.error(f"Error in interactive processor: {e}")
    
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
                
                logger.info(f"  {client_id[:20]}... -> {model_size} model (score: {performance:.1f})")
        
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
    
    logger.info("AI Load Balancer Server v1.0 started")
    logger.info(f"Listening on {listen_addr}")
    logger.info("Ready for distributed LLM processing")
    logger.info("Supported models:")
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