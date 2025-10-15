#!/usr/bin/env python3
"""
LLM Task Manager v1.0
Manages different LLM models and their deployment requirements
"""

import logging
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LLMModelConfig:
    """Configuration for LLM models"""
    model_name: str
    model_size: str  # "1B", "3B", "8B"
    huggingface_model: str
    min_ram_gb: int
    min_gpu_memory_gb: float
    recommended_cpu_cores: int
    docker_image: str
    python_script: str
    gradio_port: int
    priority: int  # Higher number = higher priority for better hardware

# LLM Model Configurations
LLM_MODELS = {
    "dhenu2-llama3.2-1b": LLMModelConfig(
        model_name="dhenu2-llama3.2-1b",
        model_size="1B",
        huggingface_model="KissanAI/Dhenu2-In-Llama3.2-1B-Instruct",
        min_ram_gb=4,
        min_gpu_memory_gb=2.0,
        recommended_cpu_cores=4,
        docker_image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
        python_script="llama1B.py",
        gradio_port=7861,
        priority=1
    ),
    "dhenu2-llama3.2-3b": LLMModelConfig(
        model_name="dhenu2-llama3.2-3b", 
        model_size="3B",
        huggingface_model="KissanAI/Dhenu2-In-Llama3.2-3B-Instruct",
        min_ram_gb=8,
        min_gpu_memory_gb=4.0,
        recommended_cpu_cores=6,
        docker_image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
        python_script="llama3B.py",
        gradio_port=7862,
        priority=2
    ),
    "dhenu2-llama3.1-8b": LLMModelConfig(
        model_name="dhenu2-llama3.1-8b",
        model_size="8B", 
        huggingface_model="KissanAI/Dhenu2-In-Llama3.1-8B-Instruct",
        min_ram_gb=16,
        min_gpu_memory_gb=8.0,
        recommended_cpu_cores=8,
        docker_image="pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime",
        python_script="llama8B.py",
        gradio_port=7863,
        priority=3
    )
}

class LLMTaskManager:
    """Manages LLM model deployment and task distribution"""
    
    def __init__(self):
        self.deployed_models: Dict[str, Dict] = {}
        self.model_assignments: Dict[str, str] = {}  # client_id -> model_name
        
    def get_optimal_model_for_client(self, client_specs: Dict) -> Optional[str]:
        """
        Determine the best LLM model for a client based on its specifications
        """
        cpu_cores = client_specs.get('cpu_cores', 0)
        ram_gb = client_specs.get('ram_gb', 0)
        gpu_memory_gb = client_specs.get('gpu_memory_gb', 0)
        performance_score = client_specs.get('performance_score', 0)
        
        logger.info(f"🔍 Finding optimal model for client:")
        logger.info(f"   CPU: {cpu_cores} cores")
        logger.info(f"   RAM: {ram_gb} GB")
        logger.info(f"   GPU: {gpu_memory_gb} GB")
        logger.info(f"   Performance: {performance_score}")
        
        # Find suitable models (meet minimum requirements)
        suitable_models = []
        
        for model_name, config in LLM_MODELS.items():
            if (cpu_cores >= config.recommended_cpu_cores and
                ram_gb >= config.min_ram_gb and
                gpu_memory_gb >= config.min_gpu_memory_gb):
                
                suitable_models.append((model_name, config))
                logger.info(f"✅ {model_name} ({config.model_size}) - SUITABLE")
            else:
                logger.info(f"❌ {model_name} ({config.model_size}) - Requirements not met")
        
        if not suitable_models:
            # Fallback to smallest model if nothing meets requirements
            logger.warning("⚠️ No models meet requirements, using smallest model")
            return "dhenu2-llama3.2-1b"
        
        # Sort by priority (highest first) and return the best one
        suitable_models.sort(key=lambda x: x[1].priority, reverse=True)
        selected_model = suitable_models[0][0]
        
        logger.info(f"🎯 Selected model: {selected_model}")
        return selected_model
    
    def distribute_models_to_clients(self, clients: Dict[str, Dict]) -> Dict[str, str]:
        """
        Intelligently distribute LLM models to clients based on their capabilities
        """
        logger.info(f"🔄 Distributing models to {len(clients)} clients")
        
        # Sort clients by performance score (highest first)
        sorted_clients = sorted(
            clients.items(), 
            key=lambda x: x[1]['client_info'].specs.performance_score, 
            reverse=True
        )
        
        # Sort models by priority (highest first)
        sorted_models = sorted(
            LLM_MODELS.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        
        assignments = {}
        used_models = set()
        
        # First pass: Assign highest priority models to best clients
        for i, (client_id, client_data) in enumerate(sorted_clients):
            client_specs = {
                'cpu_cores': client_data['client_info'].specs.cpu_cores,
                'ram_gb': client_data['client_info'].specs.ram_gb,
                'gpu_memory_gb': client_data['client_info'].specs.gpu_memory_gb,
                'performance_score': client_data['client_info'].specs.performance_score
            }
            
            # Try to assign the highest available model that fits
            for model_name, config in sorted_models:
                if model_name not in used_models:
                    if (client_specs['cpu_cores'] >= config.recommended_cpu_cores and
                        client_specs['ram_gb'] >= config.min_ram_gb and
                        client_specs['gpu_memory_gb'] >= config.min_gpu_memory_gb):
                        
                        assignments[client_id] = model_name
                        used_models.add(model_name)
                        
                        logger.info(f"✅ Assigned {model_name} ({config.model_size}) to client {client_id}")
                        logger.info(f"   Client specs: {client_specs['cpu_cores']}C/{client_specs['ram_gb']}GB/{client_specs['gpu_memory_gb']}GB GPU")
                        break
            
            # If no model assigned, give the smallest available
            if client_id not in assignments:
                for model_name, config in reversed(list(sorted_models)):
                    if model_name not in used_models:
                        assignments[client_id] = model_name
                        used_models.add(model_name)
                        logger.info(f"⚠️ Fallback: Assigned {model_name} to client {client_id}")
                        break
        
        self.model_assignments = assignments
        return assignments
    
    def get_model_deployment_config(self, model_name: str, client_id: str) -> Dict:
        """
        Get deployment configuration for a specific model
        """
        if model_name not in LLM_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = LLM_MODELS[model_name]
        
        return {
            "model_name": model_name,
            "docker_image": config.docker_image,
            "environment_vars": {
                "HUGGINGFACE_MODEL": config.huggingface_model,
                "MODEL_SIZE": config.model_size,
                "GRADIO_PORT": str(config.gradio_port),
                "CUDA_VISIBLE_DEVICES": "0"  # Use first GPU if available
            },
            "port_mappings": [
                {
                    "host_port": config.gradio_port,
                    "container_port": 7860,
                    "protocol": "tcp"
                }
            ],
            "volume_mounts": [
                {
                    "host_path": f"/tmp/{model_name}_cache",
                    "container_path": "/root/.cache",
                    "mode": "rw"
                }
            ],
            "memory_limit_mb": config.min_ram_gb * 1024,
            "cpu_limit": config.recommended_cpu_cores,
            "python_script": config.python_script,
            "gradio_port": config.gradio_port
        }
    
    def create_llm_task(self, model_name: str, prompt: str, task_id: str) -> Dict:
        """
        Create an LLM inference task
        """
        if model_name not in LLM_MODELS:
            raise ValueError(f"Unknown model: {model_name}")
        
        config = LLM_MODELS[model_name]
        
        return {
            "task_id": task_id,
            "task_type": "LLM_INFERENCE",
            "model_name": model_name,
            "prompt": prompt,
            "parameters": {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "top_p": 0.9
            },
            "endpoint_url": f"http://localhost:{config.gradio_port}",
            "expected_response_time": self._estimate_response_time(config.model_size),
            "priority": config.priority
        }
    
    def _estimate_response_time(self, model_size: str) -> float:
        """Estimate response time based on model size"""
        size_to_time = {
            "1B": 5.0,   # 5 seconds
            "3B": 15.0,  # 15 seconds  
            "8B": 45.0   # 45 seconds
        }
        return size_to_time.get(model_size, 10.0)
    
    def get_model_status_summary(self) -> Dict:
        """Get summary of model deployment status"""
        return {
            "total_models": len(LLM_MODELS),
            "deployed_models": len(self.deployed_models),
            "model_assignments": self.model_assignments,
            "available_models": list(LLM_MODELS.keys()),
            "model_configs": {
                name: {
                    "size": config.model_size,
                    "min_ram": config.min_ram_gb,
                    "min_gpu": config.min_gpu_memory_gb,
                    "priority": config.priority
                }
                for name, config in LLM_MODELS.items()
            }
        }

# Global instance
llm_task_manager = LLMTaskManager()