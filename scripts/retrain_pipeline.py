#!/usr/bin/env python3
"""
Automated Model Retraining Pipeline

Orchestrates the complete retraining workflow:
1. Data collection from Kafka logs
2. Data processing and deduplication  
3. ALS model training
4. Get PopGen models
5. Model evaluation
6. Drift detection and statistics generation
7. Provenance generation
8. Artifact packaging
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import shutil

# Add project root to path for imports before importing project modules
sys.path.insert(0, '/app')

from src.provenance.manifest import build_model_manifest, persist_manifest

def run_command(cmd, description):
    """Run a shell command and handle errors, including timeout for streaming commands."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.stdout:
        print(result.stdout)
    
    if result.returncode == 0:
        print(f"✅ {description} completed successfully")
    elif result.returncode == 124:
        print(f"⚠️  Command timed out (exit code 124) - expected for streaming")
        print(f"✅ {description} completed with timeout")
    else:
        print(f"❌ ERROR: {description} failed (exit code: {result.returncode})")
        if result.stderr:
            print(f"Error output:\n{result.stderr}")
        sys.exit(1)
    
    # Only print stderr for real errors
    if result.stderr and result.returncode not in (0, 124):
        print(f"Stderr output:\n{result.stderr}")
    
    return result

def collect_data(data_dir, data_version):
    """Collect and process training data from Kafka"""
    print(f"\n{'='*60}")
    print("STEP 1: DATA COLLECTION FROM KAFKA")
    print(f"{'='*60}")
    
    # Create dataset directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Pull logs from Kafka using kcat (only new messages since last offset)
    raw_log_path = os.path.join(data_dir, "raw.log")
    
    print(f"Pulling new logs from Kafka topic 'movielog13'...")
    cmd = f"""
    timeout 300 kcat -b localhost:9092 \
        -G movie-recommender-retraining movielog13 \
        -q > {raw_log_path}
    """
    
    run_command(cmd, "Fetching Kafka logs")
    
    # Verify we got data
    log_size = os.path.getsize(raw_log_path)
    if log_size == 0:
        print("⚠️  No new messages in Kafka since last run")
        print("   Using synthetic fallback data for testing")
        # Create minimal fallback to allow pipeline to continue
        with open(raw_log_path, 'w') as f:
            f.write("")  # Empty file, will result in no interactions
        print("   Pipeline will proceed with empty dataset")
    else:
        print(f"✅ Kafka logs fetched: {log_size / (1024**3):.2f} GB")
    
    # Process log to interactions.csv
    interactions_path = os.path.join(data_dir, "interactions.csv")
    
    # Use Keshar's parsing logic from evaluation/online_als.py
    cmd = f"""
    python3 -u -c "
import sys
sys.path.insert(0, '/app/src')
from evaluation.online_als import parse_log_file
import pandas as pd

print('Parsing Kafka logs...')
recs_by_user, watches_by_user, total_lines = parse_log_file('{raw_log_path}')

print(f'Processed {{total_lines}} log lines')
print(f'Found {{len(recs_by_user)}} users with recommendations')
print(f'Found {{len(watches_by_user)}} users with watches')

# Stream to CSV in chunks to avoid memory spike (32M+ events)
print('Writing interactions to CSV (streaming)...')
interaction_count = 0
chunk_size = 100000

with open('{interactions_path}', 'w', newline='') as f:
    import csv
    writer = csv.writer(f)
    writer.writerow(['user_id', 'movie_id', 'timestamp', 'rating'])
    
    chunk = []
    for user_id, watch_events in watches_by_user.items():
        for timestamp, movie_id in watch_events:
            # Implicit feedback: 1 = watched (binary interaction)
            chunk.append([user_id, movie_id, timestamp, 1])
            interaction_count += 1
            
            if len(chunk) >= chunk_size:
                writer.writerows(chunk)
                chunk = []
                if interaction_count % 1000000 == 0:
                    print(f'  Wrote {{interaction_count:,}} interactions...')
    
    if chunk:
        writer.writerows(chunk)

print(f'✅ Created {{interaction_count:,}} interactions')

# Clear memory before proceeding
del watches_by_user, recs_by_user
import gc
gc.collect()
"
    """
    
    run_command(cmd, "Processing Kafka logs to interactions.csv")
    
    # Deduplicate interactions
    dedup_path = os.path.join(data_dir, "interactions_dedup.csv")
    cmd = f"""
    python3 -u -c "
import pandas as pd
df = pd.read_csv('{interactions_path}')
print(f'Before dedup: {{len(df)}} interactions')
df_dedup = df.drop_duplicates(subset=['user_id', 'movie_id'], keep='last')
print(f'After dedup: {{len(df_dedup)}} interactions')
df_dedup.to_csv('{dedup_path}', index=False)
print('✅ Deduplication complete')
"
    """
    
    run_command(cmd, "Deduplicating interactions")
    
    print(f"\n✅ Data collection complete")
    print(f"   Raw log: {raw_log_path}")
    print(f"   Raw log size: {log_size / (1024**3):.2f} GB")
    print(f"   Interactions: {interactions_path}")
    print(f"   Deduplicated: {dedup_path}")
    
    return dedup_path

def train_als_model(data_path, model_dir):
    """Train ALS recommendation model"""
    print(f"\n{'='*60}")
    print("STEP 2: TRAIN ALS MODEL")
    print(f"{'='*60}")
    
    cmd = f"""
    cd /app && python3 src/models/als/train.py \
        --data_path {data_path} \
        --artifact_dir {model_dir}
    """
    
    run_command(cmd, "Training ALS model")
    
    # Verify artifacts were created
    required_files = ['als_model.npz', 'user_map.json', 'movie_map.json']
    for filename in required_files:
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ ERROR: Missing required file: {filepath}")
            sys.exit(1)
    
    print("✅ All ALS artifacts created")

def get_popgen_models(model_dir):
    """Get PopGen models from shared artifacts directory"""
    print(f"\n{'='*60}")
    print("STEP 3: GET POPGEN MODELS")
    print(f"{'='*60}")
    
    # Source directory for PopGen models
    dir = "/home/mlprod/artifacts"
    
    popgen_files = [
        'popgen_popularity.parquet',
        'popgen_user_genre_prefs.parquet'
    ]
    
    for filename in popgen_files:
        src = os.path.join(dir, filename)
        dst = os.path.join(model_dir, filename)
        
        if os.path.exists(src):
            print(f"Copying {filename}...")
            shutil.copy2(src, dst)
            print(f"  ✅ {src} -> {dst}")
        else:
            print(f"❌ ERROR: Missing popgen model file: {src}")
            print(f"   Please ensure PopGen models are in {dir}")
            sys.exit(1)
    
    print("✅ All PopGen models copied")

def evaluate_model(model_dir, data_path):
    """Evaluate model performance and apply quality gate"""
    print(f"\n{'='*60}")
    print("STEP 4: MODEL EVALUATION")
    print(f"{'='*60}")
    
    # Quality gate threshold
    HIT_AT_20_THRESHOLD = 0.30
    
    eval_script = f"""#!/usr/bin/env python3
import sys
sys.path.insert(0, '/app/src')
import json
from datetime import datetime
from models.als.model import ALSRecommender
from models.als.evaluate import sample_holdouts, evaluate_hit_at_20

# Load the trained model
print('Loading ALS model from {model_dir}...')
reco = ALSRecommender.load(artifacts_dir='{model_dir}')

# Sample holdout pairs for evaluation
print('Sampling holdout pairs...')
pairs = sample_holdouts('{data_path}', n_users=1000)

# Compute Hit@20
print('Computing Hit@20...')
hit_at_20 = evaluate_hit_at_20(reco, pairs)

print(f'Hit@20: {{hit_at_20:.4f}}')

quality_gate_passed = hit_at_20 >= {HIT_AT_20_THRESHOLD}

results = {{
    'hit_at_20': float(hit_at_20),
    'quality_gate_threshold': {HIT_AT_20_THRESHOLD},
    'quality_gate_passed': quality_gate_passed,
    'evaluation_samples': len(pairs),
    'evaluation_timestamp': datetime.now().isoformat()
}}

# Save results
with open('{model_dir}/eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
"""
    
    # Write script to temp file
    script_path = '/tmp/eval_model.py'
    with open(script_path, 'w') as f:
        f.write(eval_script)
    
    cmd = f"python3 {script_path}"
    result = run_command(cmd, "Evaluating model performance")
    
    # Read and check results
    eval_path = os.path.join(model_dir, 'eval_results.json')
    with open(eval_path, 'r') as f:
        results = json.load(f)
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Hit@20: {results['hit_at_20']:.4f}")
    print(f"Threshold: {results['quality_gate_threshold']:.4f}")
    print(f"Quality Gate: {'✅ PASSED' if results['quality_gate_passed'] else '❌ FAILED'}")
    print(f"{'='*60}")
    
    if not results['quality_gate_passed']:
        print("\n❌ Model failed quality gate - stopping deployment")
        sys.exit(1)
    
    print("\n✅ Model passed quality gate")
    return results

def check_drift_and_compute_stats(data_path, baseline_dir, model_dir):
    """
    Run check_drift.py to detect drift and compute reference statistics.

    Args:
        data_path: Current training data (interactions_dedup.csv)
        baseline_dir: Directory containing baseline/reference data
        model_dir: Directory to save drift report
    """
    print(f"\n{'='*60}")
    print("STEP 5: DRIFT DETECTION AND STATISTICS")
    print(f"{'='*60}")

    # Define paths
    baseline_path = os.path.join(baseline_dir, "interactions_dedup.csv")
    drift_report_path = os.path.join(model_dir, "drift_report.json")

    # Build check_drift.py command
    if os.path.exists(baseline_path):
        # Reference vs Current mode
        cmd = f"""
        python3 -m src.data_quality.check_drift \
            --ref {baseline_path} \
            --cur {data_path} \
            --out {drift_report_path}
        """
        mode = "reference-vs-current"
        print(f"Comparing against baseline:")
        print(f"  Baseline (--ref): {baseline_path}")
        print(f"  Current (--cur):  {data_path}")
    else:
        # Single snapshot mode (first training)
        cmd = f"""
        python3 -m src.data_quality.check_drift \
            --cur {data_path} \
            --out {drift_report_path}
        """
        mode = "single-snapshot"
        print(f"No baseline found at {baseline_path}")
        print("Computing statistics only (first training)")
        print(f"  Current (--cur): {data_path}")

    run_command(cmd, f"Running drift detection ({mode})")

    # Display drift summary
    with open(drift_report_path) as f:
        drift_report = json.load(f)

    print(f"\n{'='*60}")
    print("DRIFT DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Mode: {drift_report.get('mode', 'unknown')}")
    print(f"Drift Detected: {'⚠️  YES' if drift_report.get('drift_detected') else '✅ NO'}")

    # Show key metrics
    if 'metrics' in drift_report:
        metrics = drift_report['metrics']

        # Rating PSI (only in reference-vs-current mode)
        if 'rating_psi' in metrics:
            psi_val = metrics['rating_psi'].get('psi', 0)
            severity = metrics['rating_psi'].get('severity', 'none')
            print(f"Rating PSI: {psi_val:.4f} ({severity})")

        # Movie popularity
        if 'movie_popularity' in metrics:
            pop = metrics['movie_popularity']
            if 'cur' in pop:
                # Reference-vs-current mode
                print(f"Top 10 Share: {pop['cur'].get('top10_share', 0):.3f}")
                print(f"Gini Coefficient: {pop['cur'].get('gini_movies', 0):.3f}")
            else:
                # Single snapshot mode
                print(f"Top 10 Share: {pop.get('top10_share', 0):.3f}")
                print(f"Gini Coefficient: {pop.get('gini_movies', 0):.3f}")

    print(f"{'='*60}\n")

    if drift_report.get('drift_detected'):
        print("⚠️  WARNING: Drift detected - review drift_report.json for details")
    else:
        print("✅ No significant drift detected")

    print("✅ Drift detection and statistics complete")

def generate_provenance(model_dir, args, training_start, training_end, eval_results, data_path):
    """Generate enhanced provenance metadata with SHA256 checksums and data profiling"""
    print(f"\n{'='*60}")
    print("STEP 6: GENERATE PROVENANCE")
    print(f"{'='*60}")
    
    # Compute data profile
    print("Computing data profile...")
    import pandas as pd
    try:
        df = pd.read_csv(data_path)
        data_profile = {
            "row_count": len(df),
            "unique_users": int(df['user_id'].nunique()),
            "unique_movies": int(df['movie_id'].nunique()),
        }
        print(f"  Rows: {data_profile['row_count']}")
        print(f"  Unique users: {data_profile['unique_users']}")
        print(f"  Unique movies: {data_profile['unique_movies']}")
    except Exception as e:
        print(f"Could not compute data profile: {e}")
        data_profile = {}
    
    # Hyperparameters
    hparams = {
        "factors": 50,
        "regularization": 0.01,
        "iterations": 20,
        "alpha": 40,
        "random_state": 42
    }
    
    # Artifact paths
    artifact_paths = {
        "als_model": os.path.join(model_dir, "als_model.npz"),
        "user_map": os.path.join(model_dir, "user_map.json"),
        "movie_map": os.path.join(model_dir, "movie_map.json"),
        "popgen_popularity": os.path.join(model_dir, "popgen_popularity.parquet"),
        "popgen_user_genre_prefs": os.path.join(model_dir, "popgen_user_genre_prefs.parquet"),
        "drift_report": os.path.join(model_dir, "drift_report.json"),
        "eval_results": os.path.join(model_dir, "eval_results.json"),
    }
    
    # Build manifest
    print("Building manifest with SHA256 checksums...")
    manifest = build_model_manifest(
        model_family="als",
        artifact_paths=artifact_paths,
        hyperparameters=hparams,
        data_profile=data_profile,
        data_path=data_path,
        data_version=args.data_version,
        model_version=args.model_version,
        extra_metadata={
            "training_duration_seconds": (training_end - training_start).total_seconds(),
            "trained_at": training_start.isoformat(),
            "evaluation": eval_results,
            "environment": {
                "jenkins_build": args.build_number,
                "python_version": sys.version.split()[0],
                "training_node": os.uname().nodename,
            },
        }
    )
    
    # Persist manifest
    manifest_path = persist_manifest(manifest, model_dir, args.model_version)
    
    print(f"Provenance saved: {manifest_path}")
    print(f"  Legacy: {os.path.join(model_dir, 'provenance.json')}")
    print(f"  Versioned: {os.path.join(model_dir, 'manifests', args.model_version + '.json')}")
    print(f"\nManifest summary:")
    print(f"  Model: {manifest['model_version']}")
    pipeline_info = manifest.get('pipeline') or {}
    commit = pipeline_info.get('git_commit')
    commit_display = commit[:8] if commit else 'N/A'
    print(f"  Pipeline commit: {commit_display}")
    print(f"  Data version: {manifest['data'].get('dataset_version', 'N/A')}")
    sha = manifest.get('manifest_sha256', 'N/A')
    print(f"  Manifest SHA256: {sha[:16] + '...' if sha != 'N/A' else sha}")


def main():
    parser = argparse.ArgumentParser(description='Automated model retraining pipeline')
    parser.add_argument('--model-version', required=True, help='Model version (e.g., v20241113_020034)')
    parser.add_argument('--data-version', required=True, help='Data version (e.g., data_20241113_020034)')
    parser.add_argument('--git-commit', required=True, help='Git commit hash')
    parser.add_argument('--build-number', required=True, help='Jenkins build number')
    args = parser.parse_args()
    
    training_start = datetime.now()
    
    print("\n" + "="*60)
    print("AUTOMATED MODEL RETRAINING PIPELINE")
    print("="*60)
    print(f"Model Version: {args.model_version}")
    print(f"Data Version: {args.data_version}")
    print(f"Git Commit: {args.git_commit}")
    print(f"Build Number: {args.build_number}")
    print(f"Start Time: {training_start.isoformat()}")
    print("="*60)
    
    # Setup directories
    model_dir = f"/home/mlprod/models/{args.model_version}"
    data_dir = f"/home/mlprod/datasets/{args.data_version}"
    baseline_dir = "/home/mlprod/datasets/baseline"

    os.makedirs(model_dir, exist_ok=True)
    
    try:
        # Step 1: Collect and process data
        data_path = collect_data(data_dir, args.data_version)
        
        # Step 2: Train ALS model
        train_als_model(data_path, model_dir)
        
        # Step 3: Get PopGen models
        get_popgen_models(model_dir)
        
        # Step 4: Evaluate model
        eval_results = evaluate_model(model_dir, data_path)

        # Step 5: Check drift and compute statistics
        check_drift_and_compute_stats(data_path, baseline_dir, model_dir)

        # Step 6: Generate provenance
        training_end = datetime.now()
        generate_provenance(model_dir, args, training_start, training_end, eval_results, data_path)
        
        print(f"\n{'='*60}")
        print("✅ RETRAINING PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Model artifacts: {model_dir}")
        print(f"Training data: {data_dir}")
        print(f"Duration: {(training_end - training_start).total_seconds():.2f} seconds")
        print(f"\nArtifacts generated:")
        print(f"  - ALS model (als_model.npz, user_map.json, movie_map.json)")
        print(f"  - PopGen models (popgen_popularity.parquet, popgen_user_genre_prefs.parquet)")
        print(f"  - Drift report (drift_report.json)")
        print(f"  - Provenance metadata")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n❌ PIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
