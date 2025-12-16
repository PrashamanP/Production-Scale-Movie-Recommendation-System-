# Kubernetes Monitoring Deployment Guide

## 📁 Files in this Directory

This directory contains Kubernetes manifests for deploying the monitoring stack (Prometheus, Grafana, Alertmanager) into your K3s cluster. These files **complement** Cynthia's existing Docker Compose setup in `monitoring/`.

### Files:
- **`prometheus.yaml`** (296 lines) - Prometheus with K8s service discovery + RBAC
- **`grafana.yaml`** (138 lines) - Grafana with auto-provisioned dashboards
- **`alertmanager.yaml`** (125 lines) - Alertmanager for alert routing

### What's Reused from Cynthia's Work:
✅ Alert rules (`monitoring/prometheus_rules.yml`)  
✅ Grafana dashboards (`monitoring/grafana/dashboards/`)  
✅ Alertmanager config (`monitoring/alertmanager.yml`)  
✅ Custom metrics (`monitoring/custom_metrics.py` - already in Flask app!)

## 🚀 Quick Deployment

### For Production K3s Cluster

#### Prerequisites:
1. K3s cluster is running
2. Movie-recommender pods are deployed (3 replicas with `app=movie-recommender` label)
3. You're on the `mlprod` account

#### Step 1: Create monitoring directories
```bash
sudo mkdir -p /home/mlprod/monitoring/prometheus
sudo mkdir -p /home/mlprod/monitoring/grafana
sudo mkdir -p /home/mlprod/monitoring/alertmanager
sudo chown -R mlprod:mlprod /home/mlprod/monitoring
```

#### Step 2: Deploy monitoring stack
```bash
# Switch to mlprod account
ssh mlprod@17645-team13.isri.cmu.edu

# Navigate to repo
cd /home/mlprod/group-project-f25-blockbusters
git checkout feat-add-monitoring-m3
git pull origin feat-add-monitoring-m3

# Deploy all monitoring components
export KUBECONFIG=/etc/rancher/k3s/k3s.yaml
sudo kubectl apply -f k8s/monitoring/
```

#### Step 3: Wait for pods to be ready
```bash
sudo kubectl get pods -l 'app in (prometheus,grafana,alertmanager)' -w
```

Press `Ctrl+C` when all 3 pods show `Running` and `1/1` ready.

#### Step 4: Verify deployment
```bash
# Check all monitoring pods
sudo kubectl get pods -l 'app in (prometheus,grafana,alertmanager)'

# Check services and NodePorts
sudo kubectl get services -l 'app in (prometheus,grafana,alertmanager)'

# Verify Prometheus is scraping your app pods
curl http://localhost:30090/api/v1/targets | jq '.data.activeTargets[] | {job, instance, health}'
```

#### Step 5: Access dashboards
- **Prometheus**: http://17645-team13.isri.cmu.edu:30090
- **Grafana**: http://17645-team13.isri.cmu.edu:30030 (admin/admin)
- **Alertmanager**: http://17645-team13.isri.cmu.edu:30093

---

### For Local Minikube Testing

#### Step 1: Start Minikube
```bash
minikube start --driver=docker --memory=4096 --cpus=2
```

#### Step 2: Point Docker to Minikube's Docker Daemon
```bash
eval $(minikube docker-env)
```

#### Step 3: Build Docker Image Inside Minikube
```bash
docker build -t movie-recommender:latest .
```

#### Step 4: Deploy All Kubernetes Resources
```bash
# Deploy everything (app + monitoring) at once with --recursive flag
kubectl apply -f k8s/ --recursive

# Or deploy separately:
# kubectl apply -f k8s/deployment.yaml
# kubectl apply -f k8s/service.yaml
# kubectl apply -f k8s/pdb.yaml
# kubectl apply -f k8s/monitoring/ --recursive
```

#### Step 5: Check Pod Status (App Pods Will Fail, Monitoring Pods May Crash)
```bash
kubectl get pods
# App pods: ContainerCreating (waiting for models volume)
# Monitoring pods: CrashLoopBackOff (wrong permissions)
```

#### Step 6: Create Directories and Fix Permissions
**Note:** App needs models directory, monitoring directories are auto-created but need correct permissions.

```bash
# Create models directory and copy artifacts
minikube ssh 'sudo mkdir -p /home/mlprod/models/latest'
minikube cp artifacts/ /home/mlprod/models/latest/

# Fix monitoring permissions (directories auto-created by K8s)
# Prometheus runs as UID 65534
minikube ssh 'sudo chown -R 65534:65534 /home/mlprod/monitoring/prometheus'
minikube ssh 'sudo chmod -R 755 /home/mlprod/monitoring/prometheus'

# Grafana runs as UID 472
minikube ssh 'sudo chown -R 472:472 /home/mlprod/monitoring/grafana'
minikube ssh 'sudo chmod -R 755 /home/mlprod/monitoring/grafana'

# Alertmanager runs as UID 65534
minikube ssh 'sudo chown -R 65534:65534 /home/mlprod/monitoring/alertmanager'
minikube ssh 'sudo chmod -R 755 /home/mlprod/monitoring/alertmanager'
```

#### Step 7: Restart Failed Pods
```bash
kubectl delete pod -l app=prometheus
kubectl delete pod -l app=grafana
kubectl delete pod -l app=alertmanager
```

#### Step 8: Wait for All Pods to Be Ready
```bash
kubectl get pods -w
# Press Ctrl+C when all 6 pods show Running (3 app + 3 monitoring)
```

#### Step 9: Verify Deployment
```bash
# Check all pods
kubectl get pods

# Check services
kubectl get services

# Check Prometheus is scraping app pods
kubectl exec deployment/prometheus -- wget -qO- http://localhost:9090/api/v1/targets | grep movie-recommender
```

#### Step 10: Access Services
```bash
# Get service URLs (keep terminals open)
minikube service movie-recommender --url  # App
minikube service grafana --url           # Grafana (admin/admin)
minikube service prometheus --url        # Prometheus
minikube service alertmanager --url      # Alertmanager

# Or use port-forwarding:
kubectl port-forward svc/grafana 3000:3000 &
kubectl port-forward svc/prometheus 9090:9090 &
kubectl port-forward svc/alertmanager 9093:9093 &
kubectl port-forward svc/movie-recommender 8082:8082 &

# Access at:
# http://localhost:3000 (Grafana)
# http://localhost:9090 (Prometheus)
# http://localhost:9093 (Alertmanager)
# http://localhost:8082/recommend/123 (App)
```

#### Step 11: Test Recommendations
```bash
# Get app URL
APP_URL=$(minikube service movie-recommender --url)

# Test endpoint
curl $APP_URL/recommend/123

# Load test (generate traffic for metrics)
for i in {1..100}; do
  curl -s $APP_URL/recommend/$i > /dev/null
  echo -n "."
done
echo ""
```

#### Clean Up Minikube
```bash
# Delete cluster
minikube delete

# Start fresh
minikube start --driver=docker --memory=4096 --cpus=2
```

## 🔍 How It Works

### Kubernetes Service Discovery
Prometheus automatically discovers your movie-recommender pods using the Kubernetes API:

```yaml
kubernetes_sd_configs:
  - role: pod
relabel_configs:
  - source_labels: [__meta_kubernetes_pod_label_app]
    action: keep
    regex: movie-recommender  # Only scrape pods with this label
```

**Benefits:**
- ✅ Auto-discovers new pods when you scale up
- ✅ Removes old pods when you scale down  
- ✅ No hardcoded IPs needed
- ✅ Works with any number of replicas

### RBAC (Role-Based Access Control)
Prometheus needs permission to query the Kubernetes API:

```yaml
ServiceAccount: prometheus
ClusterRole: Read permissions for pods, services, endpoints
ClusterRoleBinding: Binds the account to the role
```

Without RBAC, Prometheus can't discover pods and will fail to scrape.

### Persistent Storage
Data is stored on the VM host using `hostPath` volumes:

```yaml
volumes:
  - name: prometheus-storage
    hostPath:
      path: /home/mlprod/monitoring/prometheus
      type: DirectoryOrCreate
```

**Benefits:**
- ✅ Data survives pod restarts
- ✅ Historical metrics preserved
- ✅ Consistent with your existing model storage pattern

## 📊 What Gets Deployed

### Prometheus
- **ConfigMap**: prometheus.yml (K8s service discovery) + prometheus_rules.yml (9 alerts)
- **ServiceAccount + RBAC**: Permissions to discover pods
- **Deployment**: 1 replica, 512Mi-1Gi memory
- **Service**: NodePort 30090
- **Scrapes**: All pods with `app=movie-recommender` label every 15s
- **Stores**: Metrics in `/home/mlprod/monitoring/prometheus`

### Grafana
- **ConfigMaps**: Datasource (Prometheus) + Dashboard provider
- **Deployment**: 1 replica, 256Mi-512Mi memory
- **Service**: NodePort 30030
- **Auto-loads**: 11-panel "Recommendation Service Dashboard"
- **Stores**: Settings in `/home/mlprod/monitoring/grafana`

### Alertmanager  
- **ConfigMap**: alertmanager.yml (alert routing)
- **Deployment**: 1 replica, 128Mi-256Mi memory
- **Service**: NodePort 30093
- **Routes**: 9 alert rules from Prometheus
- **Stores**: Alert state in `/home/mlprod/monitoring/alertmanager`

## 🧪 Testing & Verification

### Generate test traffic:
```bash
# Single request
curl http://17645-team13.isri.cmu.edu:30082/recommend/123

# Load test (100 requests)
for i in {1..100}; do curl -s http://localhost:30082/recommend/$i > /dev/null; echo -n "."; done
echo ""
```

### Check metrics are updating:
```bash
# Check Hit@20 metric
curl 'http://localhost:30090/api/v1/query?query=model_hit_at_20' | jq

# Check request rate
curl 'http://localhost:30090/api/v1/query?query=rate(flask_http_request_total[1m])' | jq

# Check if any alerts are firing
curl http://localhost:30090/api/v1/alerts | jq '.data.alerts[] | {alertname, state}'
```

### Verify service discovery is working:
```bash
# Scale up to 5 pods
sudo kubectl scale deployment movie-recommender --replicas=5

# Check Prometheus discovered new pods (wait 30s)
sleep 30
curl http://localhost:30090/api/v1/targets | jq '.data.activeTargets | length'
# Should show 5 targets

# Scale back down
sudo kubectl scale deployment movie-recommender --replicas=3
```

## 🐛 Troubleshooting

### Pods not starting?
```bash
# Check pod status
sudo kubectl get pods -l app=prometheus

# Check why pod is failing
sudo kubectl describe pod <pod-name>

# Check logs
sudo kubectl logs -l app=prometheus --tail=50
```

### Prometheus not scraping pods?
```bash
# Check RBAC is configured
sudo kubectl get serviceaccount prometheus
sudo kubectl get clusterrole prometheus
sudo kubectl get clusterrolebinding prometheus

# Check Prometheus config loaded correctly
sudo kubectl get configmap prometheus-config -o yaml

# Check Prometheus logs for errors
sudo kubectl logs -l app=prometheus | grep -i error
```

### Grafana dashboard not loading?
```bash
# Check dashboard file is accessible
sudo kubectl exec deployment/grafana -- ls -la /etc/grafana/provisioning/dashboards/

# Check Grafana logs
sudo kubectl logs -l app=grafana --tail=50

# Verify datasource is configured
curl -u admin:admin http://localhost:30030/api/datasources | jq
```

### Metrics not showing in Grafana?
```bash
# Check Prometheus is reachable from Grafana pod
sudo kubectl exec deployment/grafana -- wget -O- http://prometheus:9090/-/healthy

# Check /metrics endpoint on app pods
curl http://localhost:30082/metrics | grep model_hit_at_20
```

## 🔄 Updating Configuration

### Update Prometheus scrape config:
```bash
# Edit k8s/monitoring/prometheus.yaml
# Then apply changes:
sudo kubectl apply -f k8s/monitoring/prometheus.yaml

# Reload Prometheus config (hot reload, no restart needed):
sudo kubectl exec deployment/prometheus -- kill -HUP 1
```

### Update alert rules:
```bash
# Edit k8s/monitoring/prometheus.yaml (find prometheus_rules.yml section)
# Apply changes:
sudo kubectl apply -f k8s/monitoring/prometheus.yaml
sudo kubectl exec deployment/prometheus -- kill -HUP 1
```

### Update Grafana dashboard:
```bash
# Option 1: Edit monitoring/grafana/dashboards/recommendation_service.json
# Grafana will auto-reload every 10 seconds

# Option 2: Use Grafana UI to edit, then export and save to repo
```

## 🗑️ Uninstalling

To remove the monitoring stack:
```bash
sudo kubectl delete -f k8s/monitoring/

# Optionally, remove persistent data:
sudo rm -rf /home/mlprod/monitoring/
```

## 📚 Key Differences from Docker Compose

| Aspect | Docker Compose | Kubernetes |
|--------|---------------|------------|
| **Scrape targets** | `flask_app:8082` | K8s service discovery |
| **Discovery** | Static DNS | Dynamic pod labels |
| **RBAC** | Not needed | Required (ServiceAccount) |
| **Networking** | Bridge network | K8s Services + NodePorts |
| **Storage** | Docker volumes | hostPath volumes |
| **Scaling** | Manual restart | Auto-discovers new pods |

## 🎯 M3 Demo Checklist

- [ ] All 6 pods running (3 app + 3 monitoring)
- [ ] Prometheus shows 3 healthy targets
- [ ] Grafana dashboard loads with 11 panels
- [ ] Metrics update after generating traffic
- [ ] Alerts visible in Prometheus UI
- [ ] Can explain K8s service discovery
- [ ] Can show auto-scaling discovery (optional)

## 📖 Additional Resources

- **Cynthia's Docker setup**: `monitoring/README.md`
- **Integration analysis**: `docs/MONITORING-INTEGRATION-ANALYSIS.md`
- **Prometheus K8s docs**: https://prometheus.io/docs/prometheus/latest/configuration/configuration/#kubernetes_sd_config
- **Grafana provisioning**: https://grafana.com/docs/grafana/latest/administration/provisioning/

---

**Questions?** This setup is ready to deploy! Just follow the Quick Deployment steps above. 🚀
