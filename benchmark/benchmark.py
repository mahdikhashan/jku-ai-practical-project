import sys
import json
import uuid
import argparse
from datetime import datetime
from celery.result import AsyncResult
from celery_app import app
import tasks

def submit(args):
    """
    Submit a benchmark job
    """
    job_id = str(uuid.uuid4())[:8]
    
    if args.code_file:
        with open(args.code_file, 'r') as f:
            code = f.read()
    else:
        print("Enter your code (Ctrl+D when done):")
        code = sys.stdin.read()
    
    params = {}
    if args.batch_size:
        params['batch_size'] = args.batch_size
    if args.seq_length:
        params['seq_length'] = args.seq_length
    if args.num_iterations:
        params['num_iterations'] = args.num_iterations
    
    # task
    task_map = {
        'memory': tasks.benchmark_memory,
        'throughput': tasks.benchmark_throughput,
        'inference': tasks.benchmark_inference,
        'stability': tasks.benchmark_stability,
    }
    
    task = task_map[args.type]
    result = task.delay(job_id, code, params)
    
    print(f"✓ Job submitted: {job_id}")
    print(f"  Task ID: {result.id}")
    print(f"  Type: {args.type}")
    print(f"  Status: PENDING")
    print(f"\nCheck status with: python benchmark.py status {result.id}")

def status(args):
    """
    Check job status
    """
    result = AsyncResult(args.task_id, app=app)
    
    print(f"Task ID: {args.task_id}")
    print(f"Status: {result.state}")
    
    if result.state == 'PENDING':
        print("  (Job is queued or worker not started)")
    elif result.state == 'STARTED':
        print("  (Job is running...)")
    elif result.state == 'SUCCESS':
        print(f"\n✓ Results:")
        print(json.dumps(result.result, indent=2))
    elif result.state == 'FAILURE':
        print(f"\n✗ Error:")
        print(str(result.info))

def list_jobs(args):
    """List all jobs from results.json"""
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
        
        if not results:
            print("No jobs found")
            return
        
        print(f"{'Job ID':<12} {'Type':<12} {'Status':<12} {'Device':<8} {'Timestamp':<20}")
        print("-" * 80)
        
        for r in results[-args.limit:]:
            job_id = r.get('job_id', 'N/A')
            job_type = r.get('type', 'N/A')
            status = r.get('status', 'N/A')
            device = r.get('device', 'N/A')
            timestamp = r.get('timestamp', 'N/A')[:19]
            
            print(f"{job_id:<12} {job_type:<12} {status:<12} {device:<8} {timestamp:<20}")
            
    except FileNotFoundError:
        print("No results file found. Submit a job first.")

def view(args):
    """View detailed results for a job"""
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
        
        job = next((r for r in results if r['job_id'] == args.job_id), None)
        
        if not job:
            print(f"Job {args.job_id} not found")
            return
        
        print("=" * 60)
        print(f"Job: {job['job_id']}")
        print("=" * 60)
        print(f"Type: {job.get('type')}")
        print(f"Status: {job.get('status')}")
        print(f"Device: {job.get('device')}")
        print(f"Timestamp: {job.get('timestamp')}")
        print()
        
        if job['status'] == 'failed':
            print("Error:")
            print(job.get('error'))
            if args.verbose:
                print("\nTraceback:")
                print(job.get('traceback'))
        else:
            print("Results:")
            for key, value in job.items():
                if key not in ['job_id', 'type', 'status', 'device', 'timestamp']:
                    print(f"  {key}: {value}")
        
        print("=" * 60)
        
    except FileNotFoundError:
        print("No results file found")

def compare(args):
    """
    Compare two jobs
    """
    try:
        with open('results.json', 'r') as f:
            results = json.load(f)
        
        job1 = next((r for r in results if r['job_id'] == args.job1), None)
        job2 = next((r for r in results if r['job_id'] == args.job2), None)
        
        if not job1 or not job2:
            print("One or both jobs not found")
            return
        
        print("=" * 60)
        print(f"Comparing: {args.job1} vs {args.job2}")
        print("=" * 60)
        
        metrics1 = {k: v for k, v in job1.items() if isinstance(v, (int, float))}
        metrics2 = {k: v for k, v in job2.items() if isinstance(v, (int, float))}
        
        common_keys = set(metrics1.keys()) & set(metrics2.keys())
        
        if not common_keys:
            print("No common metrics to compare")
            return
        
        print(f"{'Metric':<30} {'Job 1':<15} {'Job 2':<15} {'Diff':<15}")
        print("-" * 80)
        
        for key in sorted(common_keys):
            val1 = metrics1[key]
            val2 = metrics2[key]
            diff = val2 - val1
            percent = (diff / val1 * 100) if val1 != 0 else 0
            
            print(f"{key:<30} {val1:<15.4f} {val2:<15.4f} {diff:<15.4f} ({percent:+.1f}%)")
        
        print("=" * 60)
        
    except FileNotFoundError:
        print("No results file found")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark CLI - Submit and view PyTorch/NumPy benchmarks'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    submit_parser = subparsers.add_parser('submit', help='Submit a benchmark job')
    submit_parser.add_argument('type', choices=['memory', 'throughput', 'inference', 'stability'])
    submit_parser.add_argument('-f', '--code-file', help='Python file to benchmark')
    submit_parser.add_argument('-b', '--batch-size', type=int, help='Batch size')
    submit_parser.add_argument('-s', '--seq-length', type=int, help='Sequence length')
    submit_parser.add_argument('-n', '--num-iterations', type=int, help='Number of iterations')
    submit_parser.set_defaults(func=submit)
    
    status_parser = subparsers.add_parser('status', help='Check job status')
    status_parser.add_argument('task_id', help='Celery task ID')
    status_parser.set_defaults(func=status)
    
    list_parser = subparsers.add_parser('list', help='List all jobs')
    list_parser.add_argument('-l', '--limit', type=int, default=20, help='Number of jobs to show')
    list_parser.set_defaults(func=list_jobs)
    
    view_parser = subparsers.add_parser('view', help='View job details')
    view_parser.add_argument('job_id', help='Job ID')
    view_parser.add_argument('-v', '--verbose', action='store_true', help='Show full traceback on errors')
    view_parser.set_defaults(func=view)
    
    compare_parser = subparsers.add_parser('compare', help='Compare two jobs')
    compare_parser.add_argument('job1', help='First job ID')
    compare_parser.add_argument('job2', help='Second job ID')
    compare_parser.set_defaults(func=compare)
    
    args = parser.parse_args()
        
    args.func(args)
