import time
import json
import traceback
from datetime import datetime
import torch
import numpy as np
from celery_app import app

def get_device():
    """Detect available device"""
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    else:
        return torch.device("cpu"), "cpu"

@app.task(bind=True, name='benchmark.memory')
def benchmark_memory(self, job_id, code, params):
    """Measure memory usage"""
    device, device_name = get_device()
    
    try:
        # Prepare namespace
        namespace = {
            'torch': torch,
            'np': np,
            'numpy': np,
            'device': device,
            'params': params,
        }
        
        # Execute user code
        exec(code, namespace)
        
        # Measure memory
        if device_name == "cuda":
            torch.cuda.synchronize()
            memory_mb = torch.cuda.max_memory_allocated() / 1e6
        elif device_name == "mps":
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1e6
        else:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / 1e6
        
        result = {
            'job_id': job_id,
            'type': 'memory',
            'status': 'completed',
            'device': device_name,
            'memory_mb': memory_mb,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save result
        save_result(result)
        return result
        
    except Exception as e:
        error_result = {
            'job_id': job_id,
            'type': 'memory',
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        save_result(error_result)
        raise

@app.task(bind=True, name='benchmark.throughput')
def benchmark_throughput(self, job_id, code, params):
    """Measure throughput (iterations/second)"""
    device, device_name = get_device()
    
    try:
        namespace = {
            'torch': torch,
            'np': np,
            'numpy': np,
            'device': device,
            'params': params,
        }
        
        exec(code, namespace)
        
        # User must define these in their code
        model = namespace.get('model')
        forward_fn = namespace.get('forward_fn')
        data_fn = namespace.get('data_fn')
        
        if not all([model, forward_fn, data_fn]):
            raise ValueError("Code must define: model, forward_fn, data_fn")
        
        # Warmup
        for _ in range(5):
            data = data_fn()
            forward_fn(model, data)
        
        # Benchmark
        num_iterations = params.get('num_iterations', 50)
        
        if device_name == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        for _ in range(num_iterations):
            data = data_fn()
            forward_fn(model, data)
        
        if device_name == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        throughput = num_iterations / elapsed
        
        result = {
            'job_id': job_id,
            'type': 'throughput',
            'status': 'completed',
            'device': device_name,
            'throughput_iter_per_sec': throughput,
            'time_per_iter_ms': (elapsed / num_iterations) * 1000,
            'total_time_sec': elapsed,
            'num_iterations': num_iterations,
            'timestamp': datetime.now().isoformat()
        }
        
        save_result(result)
        return result
        
    except Exception as e:
        error_result = {
            'job_id': job_id,
            'type': 'throughput',
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        save_result(error_result)
        raise

@app.task(bind=True, name='benchmark.inference')
def benchmark_inference(self, job_id, code, params):
    """Measure inference latency"""
    device, device_name = get_device()
    
    try:
        namespace = {
            'torch': torch,
            'np': np,
            'numpy': np,
            'device': device,
            'params': params,
        }
        
        exec(code, namespace)
        
        model = namespace.get('model')
        data_fn = namespace.get('data_fn')
        
        if not all([model, data_fn]):
            raise ValueError("Code must define: model, data_fn")
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                data = data_fn()
                _ = model(data)
        
        # Benchmark
        num_iterations = params.get('num_iterations', 100)
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                data = data_fn()
                
                if device_name == "cuda":
                    torch.cuda.synchronize()
                
                start = time.perf_counter()
                output = model(data)
                
                if device_name == "cuda":
                    torch.cuda.synchronize()
                
                latency_ms = (time.perf_counter() - start) * 1000
                latencies.append(latency_ms)
        
        import statistics
        result = {
            'job_id': job_id,
            'type': 'inference',
            'status': 'completed',
            'device': device_name,
            'mean_latency_ms': statistics.mean(latencies),
            'median_latency_ms': statistics.median(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        save_result(result)
        return result
        
    except Exception as e:
        error_result = {
            'job_id': job_id,
            'type': 'inference',
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        save_result(error_result)
        raise

@app.task(bind=True, name='benchmark.stability')
def benchmark_stability(self, job_id, code, params):
    """Test numerical stability across precisions"""
    device, device_name = get_device()
    
    try:
        namespace = {
            'torch': torch,
            'np': np,
            'numpy': np,
            'device': device,
            'params': params,
        }
        
        exec(code, namespace)
        
        model_fn = namespace.get('model_fn')  # Function that creates model
        test_input = namespace.get('test_input')
        
        if not all([model_fn, test_input]):
            raise ValueError("Code must define: model_fn, test_input")
        
        # Test different precisions
        precisions = ['float32', 'float16']
        if device_name == "cuda":
            precisions.append('bfloat16')
        
        outputs = {}
        for precision in precisions:
            model = model_fn().to(device)
            data = test_input.clone().to(device)
            
            if precision == 'float16':
                model = model.half()
                data = data.half()
            elif precision == 'bfloat16':
                model = model.bfloat16()
                data = data.bfloat16()
            
            output = model(data)
            outputs[precision] = output.float().cpu()
        
        # Calculate differences from float32 baseline
        import torch.nn.functional as F
        
        differences = {}
        ref_output = outputs['float32']
        
        for precision in precisions[1:]:
            mse = F.mse_loss(outputs[precision], ref_output).item()
            max_diff = (outputs[precision] - ref_output).abs().max().item()
            
            differences[precision] = {
                'mse': mse,
                'max_diff': max_diff
            }
        
        result = {
            'job_id': job_id,
            'type': 'stability',
            'status': 'completed',
            'device': device_name,
            'differences': differences,
            'timestamp': datetime.now().isoformat()
        }
        
        save_result(result)
        return result
        
    except Exception as e:
        error_result = {
            'job_id': job_id,
            'type': 'stability',
            'status': 'failed',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }
        save_result(error_result)
        raise

def save_result(result):
    """Save result to JSON file"""
    import os
    import json
    
    results_file = 'results.json'
    
    # Load existing results
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            try:
                results = json.load(f)
            except:
                results = []
    else:
        results = []
    
    # Append new result
    results.append(result)
    
    # Save back
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)