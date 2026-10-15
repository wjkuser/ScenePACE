"""Local TRELLIS worker exposing the two-phase API expected by SAGE."""
import argparse
import gc
import io
import os
import random
import tempfile
import threading
import time
import traceback
import uuid
os.environ.setdefault('ATTN_BACKEND', 'xformers')
os.environ.setdefault('SPCONV_ALGO', 'native')
import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=10070)
parser.add_argument('--model-path', default=os.environ.get('TRELLIS_MODEL_PATH', 'models/TRELLIS-text-xlarge'))
args = parser.parse_args()
print(f'Loading TRELLIS from {args.model_path}', flush=True)
pipeline = TrellisTextTo3DPipeline.from_pretrained(args.model_path)
pipeline.cuda()
print('TRELLIS pipeline loaded', flush=True)
app = Flask(__name__)
CORS(app)
jobs = {}
jobs_lock = threading.Lock()
generation_lock = threading.Lock()

def _generate(job_id, prompt, seed):
    temp_path = None
    try:
        with generation_lock:
            started_at = time.time()
            outputs = pipeline.run(prompt, seed=seed)
            glb = postprocessing_utils.to_glb(outputs['gaussian'][0], outputs['mesh'][0], simplify=0.95, texture_size=1024)
            with tempfile.NamedTemporaryFile(suffix='.glb', delete=False) as output:
                temp_path = output.name
            glb.export(temp_path)
            with open(temp_path, 'rb') as output:
                content = output.read()
        with jobs_lock:
            jobs[job_id] = {'status': 'completed', 'content': content, 'seed': seed, 'generation_time': time.time() - started_at}
    except Exception as exc:
        traceback.print_exc()
        with jobs_lock:
            jobs[job_id] = {'status': 'failed', 'error': str(exc)}
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        gc.collect()
        torch.cuda.empty_cache()

@app.get('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'sage-trellis', 'gpu_available': torch.cuda.is_available()})

@app.post('/generate')
def generate():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get('input_text', 'A simple 3D object')).strip()
    seed = int(data.get('seed', random.randint(1, 1000000)))
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {'status': 'processing'}
    threading.Thread(target=_generate, args=(job_id, prompt, seed), daemon=True).start()
    return (jsonify({'status': 'accepted', 'job_id': job_id}), 202)

@app.get('/job/<job_id>')
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return (jsonify({'error': 'Job not found'}), 404)
    if job['status'] == 'processing':
        return (jsonify({'status': 'processing', 'job_id': job_id}), 202)
    if job['status'] == 'failed':
        return (jsonify(job), 500)
    with jobs_lock:
        jobs.pop(job_id, None)
    return send_file(io.BytesIO(job['content']), as_attachment=True, download_name=f"generated_model_{job['seed']}.glb", mimetype='application/octet-stream')
if __name__ == '__main__':
    app.run(host=os.environ.get('SAGE_SERVICE_HOST', '127.0.0.1'), port=args.port, threaded=True)
