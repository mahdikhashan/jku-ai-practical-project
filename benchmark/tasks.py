from celery_app import app


@app.task(bind=True, name='benchmark.throughput')
def benchmark_throughput(self, job_id, code, params):
    """
    Measure throughput (iterations/second)
    """

    from datetime import datetime
    import time
    import traceback

    import torch


    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        elif torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        else:
            return torch.device("cpu"), "cpu"


    device, device_name = get_device()
    
    try:
        namespace = {
            'torch': torch,
            'device': device,
            'params': params,
        }
        
        exec(code, namespace)
        
        model = namespace.get('model')
        forward_fn = namespace.get('forward_fn')
        data_fn = namespace.get('data_fn')
        
        if not all([model, forward_fn, data_fn]):
            raise ValueError("Code must define: model, forward_fn, data_fn")
        
        # warmup
        for _ in range(5):
            data = data_fn()
            forward_fn(model, data)
        
        # benchmark
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

def save_result(result):
    """Save result to JSON file"""
    import os
    import json
    
    results_file = 'results.json'
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            try:
                results = json.load(f)
            except:
                results = []
    else:
        results = []
    
    results.append(result)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
