## 🤖 Automated Retraining - Complete File Package

### 📦 Files Created:

**1. [Jenkinsfile.retrain](file:///tmp/Jenkinsfile.retrain)** (root directory)
**2. [k8s/retrain-job.yaml](file:///tmp/retrain-job.yaml)** 
**3. [scripts/retrain_pipeline.py](file:///tmp/retrain_pipeline.py)**

---

## 🔄 How It Works:

### **Jenkinsfile.retrain** - Orchestrator
- **Trigger**: Runs every 3 days at 2 AM (`cron: 0 2 */3 * *`)
- **Creates**: Timestamped versions (e.g., `v20241113_020034`)
- **Stages**:
  1. Launch K8s training job
  2. Wait for training to complete (max 30 min)
  3. Check evaluation results (quality gate)
  4. Update `latest` symlink if passed
  5. Trigger k8s rolling restart
  6. Update provenance with deployment info
  7. Cleanup old datasets (keep last 5)

### **retrain-job.yaml** - K8s Training Job
- **Runs**: `retrain_pipeline.py` in a container
- **Mounts**: `/home/mlprod/models/` and `/home/mlprod/datasets/`
- **Resources**: 1-2GB RAM, 1-2 CPU cores
- **Environment**: MODEL_VERSION, DATA_VERSION, GIT_COMMIT, BUILD_NUMBER

### **retrain_pipeline.py** - Training Orchestration
**5 Steps:**
1. **Data Collection**: Parse Kafka logs → `interactions.csv`
2. **ALS Training**: Calls `src/models/als/train.py`
3. **PopGen Training**: Calls `src/models/pop-gen/train_popgen.py`
4. **Evaluation**: Computes Hit@20, applies quality gate (≥0.30)
5. **Provenance**: Generates complete metadata JSON

**Outputs**:
```
/home/mlprod/models/v20241113_020034/
├── als_model.npz
├── user_map.json
├── movie_map.json
├── popgen_popularity.parquet
├── popgen_user_genre_prefs.parquet
├── eval_results.json
└── provenance.json
```

---

## 🚀 Setup Instructions for Keshar:

### Step 1: Create PR with these 3 files
```bash
# On dballuff account
cd ~/group-project-f25-blockbusters
git checkout main
git pull origin main
git checkout -b keshar/automated-retraining

# Copy files to correct locations
cp /path/to/Jenkinsfile.retrain ./Jenkinsfile.retrain
cp /path/to/retrain-job.yaml ./k8s/retrain-job.yaml
cp /path/to/retrain_pipeline.py ./scripts/retrain_pipeline.py
chmod +x ./scripts/retrain_pipeline.py

# Commit
git add Jenkinsfile.retrain k8s/retrain-job.yaml scripts/retrain_pipeline.py
git commit -m "feat: add automated model retraining pipeline

- Jenkinsfile.retrain: Jenkins pipeline with cron trigger
- k8s/retrain-job.yaml: K8s Job template for training
- scripts/retrain_pipeline.py: Training orchestration script

Implements M3 automated retraining requirement:
- Runs every 3 days
- Quality gate (Hit@20 ≥ 0.30)
- Full provenance tracking
- Zero-downtime deployment"

git push origin keshar/automated-retraining

# Create PR
gh pr create --title "Add automated model retraining pipeline" \
  --body "See commit message for details"
```

### Step 2: After PR Merges - Setup Jenkins

**On Jenkins (http://17645-team13.isri.cmu.edu:8080):**

1. **Create New Job**:
   - Click "New Item"
   - Name: `model-retraining`
   - Type: "Pipeline"
   - Click OK

2. **Configure Job**:
   - **Build Triggers**: Check "Build periodically"
     - Schedule: `0 2 */3 * *` (every 3 days at 2 AM)
   
   - **Pipeline**:
     - Definition: "Pipeline script from SCM"
     - SCM: Git
     - Repository URL: `https://github.com/cmu-seai/group-project-f25-blockbusters.git`
     - Branch: `*/main`
     - Script Path: `Jenkinsfile.retrain`
   
   - Click "Save"

3. **Test with Manual Run**:
   - Click "Build Now"
   - Watch console output
   - Verify model created: `ssh mlprod@vm "ls -lt /home/mlprod/models/"`

### Step 3: Verify Automated System

After first successful run:
```bash
# Check new model version created
ssh mlprod@17645-team13.isri.cmu.edu "ls -lt /home/mlprod/models/"

# Check symlink updated
ssh mlprod@17645-team13.isri.cmu.edu "readlink /home/mlprod/models/latest"

# Check provenance
ssh mlprod@17645-team13.isri.cmu.edu "cat /home/mlprod/models/latest/provenance.json | jq"

# Check pods restarted
ssh mlprod@17645-team13.isri.cmu.edu "export KUBECONFIG=/etc/rancher/k3s/k3s.yaml && kubectl get pods -l app=movie-recommender"

# Test recommendations work
curl http://17645-team13.isri.cmu.edu:30082/recommend/12345
```

---

## ⚙️ Configuration Options:

### Adjust Retraining Frequency
Edit `Jenkinsfile.retrain` line 6:
```groovy
cron('0 2 */3 * *')  // Every 3 days at 2 AM

// Examples:
cron('0 2 * * *')    // Daily at 2 AM
cron('0 2 */7 * *')  // Weekly
cron('0 2 1 * *')    // Monthly on 1st
```

### Adjust Quality Gate Threshold
Edit `scripts/retrain_pipeline.py` line 197:
```python
HIT_AT_20_THRESHOLD = 0.30  # Change this value
```

### Adjust Resource Limits
Edit `k8s/retrain-job.yaml` lines 54-59:
```yaml
resources:
  requests:
    memory: "1Gi"   # Increase if OOM
    cpu: "1000m"
  limits:
    memory: "2Gi"
    cpu: "2000m"
```

---

## 🎯 What This Achieves:

✅ **Fully Automated Retraining**
- No manual intervention needed
- Runs on schedule

✅ **Quality Gates**
- Bad models never reach production
- Automatic rollback if quality drops

✅ **Full Provenance**
- Every model traceable to source data
- Git commit, training metrics, deployment time

✅ **Zero Downtime**
- Symlink atomic swap
- Rolling restart of pods

✅ **Production Grade**
- Proper error handling
- Cleanup of old data
- Jenkins integration

This satisfies M3 automated retraining requirements! 🚀
