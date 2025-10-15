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
from typing import Dict, List, Optional
import psutil
import socket

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
            self.clients[client_id] = {
                "client_info": request,
                "last_heartbeat": time.time(),
                "status": "active",
                "registered_at": time.time()
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
    
    try:
        while True:
            time.sleep(86400)  # Sleep for a day
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down server...")
        server.stop(0)

if __name__ == '__main__':
    serve()