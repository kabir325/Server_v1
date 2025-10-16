# AI Load Balancer Server (Raspberry Pi)

## What This Does
This is the **main server** that runs on your Raspberry Pi. It receives agricultural questions and distributes them to multiple laptop clients (or simulates them locally using Ollama models).

## How to Run
```bash
cd Server_v1
python ai_load_balancer_server.py
```

## What Happens Step by Step

### 1. **Server Startup**
- ✅ Loads supported LLM models configuration
- ✅ Starts gRPC server on port 50051
- ✅ Shows "Ready for distributed LLM processing"
- ✅ Waits for laptop clients to connect

### 2. **Client Registration** (when laptops connect)
- 📱 Laptop sends system specs (CPU, RAM, GPU, performance score)
- 🎯 Server assigns LLM model based on laptop performance:
  - **High performance (80+)**: 8B model
  - **Medium performance (40+)**: 3B model  
  - **Low performance (<40)**: 1B model
- ✅ Client registered and ready

### 3. **Query Processing** (when you type a question)
- 📝 You enter agricultural question (e.g., "What is fertilizer?")
- 🔄 Server shows: "PROCESSING QUERY: 'what is fertilizer?'"

### 4. **Model Distribution**
- 📊 Shows which laptop gets which model:
  ```
  Client: laptop-001 -> 3B model (performance: 85.5)
  Client: laptop-002 -> 1B model (performance: 45.2)
  ```

### 5. **Sending Queries**
- 📤 Sends same question to all connected laptops
- ⏱️ Each laptop processes with their assigned model
- 📥 Collects responses from all laptops

### 6. **Response Collection**
- ✅ "Received response from laptop-001 (3.2s)"
- ✅ "Received response from laptop-002 (1.8s)"
- 📝 Shows preview of each response

### 7. **Summarization**
- 🤖 Picks the best laptop (highest performance) for summarization
- 📋 Sends all responses to best laptop to create final summary
- ✅ Returns combined, coherent answer

### 8. **Final Result**
- 🎉 Shows complete answer combining insights from all models
- ⏱️ Shows total processing time
- 📊 Ready for next question

## Current Mode: **SIMULATION**
Since you may not have multiple laptops connected, the server simulates 2 laptop clients using local Ollama models:
- **Gaming Laptop**: Uses `llama3.2:3b` model
- **Work Laptop**: Uses `llama3.2:1b` model

## Requirements
- Python 3.7+
- gRPC libraries
- Ollama (for simulation mode)
- Models: `llama3.2:1b` and `llama3.2:3b`

## Logs You'll See
```
[SERVER DATA FLOW] STARTING DISTRIBUTED QUERY PROCESSING
[SERVER DATA FLOW] MODEL DISTRIBUTION
[SERVER DATA FLOW] SENDING QUERIES TO CLIENTS  
[OLLAMA CALL] Calling local Ollama model
[LOCAL SUMMARIZATION] SUMMARIZING RESPONSES
[SERVER DATA FLOW] FINAL RESULT
```

The server handles everything automatically - just type your agricultural questions and see the distributed AI in action! 🚀