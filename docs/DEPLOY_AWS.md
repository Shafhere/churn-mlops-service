# AWS Deployment Guide — Single EC2 Instance + Docker Compose + Caddy HTTPS

**What this is:** a deployment plan and runnable configuration for a single EC2 Linux VM. Hosting has **not** been performed by the package author, and the final public URL cannot be known before AWS resources and DNS exist. The supplied app/model code is ready for the deployment steps below; AWS account actions remain with the account owner.

## Required external resources

- An AWS account with permission to create EC2 instances, security groups, EBS storage and optionally Elastic IP. Confirm current costs and create a budget alarm before launch. EC2, public IPv4, storage and network transfer may all incur costs.
- Ubuntu 24.04 LTS EC2 instance (suggested **2 vCPU / 4 GiB RAM or better** for a combined API/UI demo; measure actual usage). At least 20–30 GiB EBS recommended; these are planning estimates, not load-tested minimums.
- A **public DNS name** under your control pointed to the instance public/Elastic IP. Caddy obtains an HTTPS certificate once DNS resolves and ports 80/443 are reachable.
- Your SSH private key file. Never add it to this project or send it in chat.

Security group inbound rules: **22/TCP only from your IP**, **80/TCP and 443/TCP** as required for public HTTPS; *do not open 8000, 8501 or 5000*.

## 1. Install Docker on Ubuntu

SSH into EC2 as `ubuntu`, then use the **official Docker Engine Ubuntu apt repository installation procedure**: https://docs.docker.com/engine/install/ubuntu/ . Install Docker Engine, the Buildx plugin and the Docker Compose plugin. Confirm:

```bash
sudo docker --version
sudo docker compose version
```

For convenience you can run the Compose commands below with `sudo docker compose` and leave the `ubuntu` user outside the privileged Docker group. Membership in the `docker` group grants powerful host privileges.

## 2. Transfer this release

Unzip the **final release** on your local Windows machine (or transfer the ZIP using `scp`) and copy it to the EC2 instance. Example from Windows PowerShell in the folder with the ZIP:

```powershell
scp -i "PATH_TO_YOUR_SSH_KEY.pem" "ChurnGuard_Final_Deployment_Ready.zip" ubuntu@YOUR_EC2_PUBLIC_IP:~/
```

On EC2:

```bash
sudo apt-get update && sudo apt-get install -y unzip python3
unzip ChurnGuard_Final_Deployment_Ready.zip
cd ChurnGuard_Final_Ready
python3 scripts/create_env.py
```

This generates random UI and API credentials in `.env` (private to this server). Add your DNS name to that file:

```bash
nano .env
```

Add exactly one line such as:

```dotenv
CHURNGUARD_DOMAIN=churn.example.com
```

Replace `churn.example.com` with your real DNS name. Configure an `A` record pointing to your instance public IP. Do **not** point it to localhost or enter the IP as a bare HTTPS domain. Use a stable IP if the demo must survive an EC2 stop/start.

## 3. Start the HTTPS application

In the project directory:

```bash
sudo docker compose -f docker-compose.yml -f deploy/docker-compose.aws.yml config --quiet
sudo docker compose -f docker-compose.yml -f deploy/docker-compose.aws.yml up -d --build
sudo docker compose -f docker-compose.yml -f deploy/docker-compose.aws.yml ps
```

Caddy serves `https://YOUR_DOMAIN` → Streamlit privately on port 8501 → FastAPI privately on port 8000. Only Caddy is exposed on public ports 80/443. HTTPS certificate provisioning needs correct DNS, allowed ports and a reachable server. Caddy's certificate storage is persisted in a named volume.

## 4. Verify before sharing the public link

- Open `https://YOUR_DOMAIN` from a device outside EC2. Confirm no TLS certificate warning and sign in with the UI password from EC2 `.env`.
- Run a **single prediction** and a **batch CSV prediction** with the public demo dataset.
- Open Monitoring and verify request count, prediction count, latency and PSI display.
- Confirm the cloud security group does **not** expose ports 8000, 8501, MLflow port 5000 or SQLite.
- Confirm failing authentication and incorrect input produce HTTP 401 and 422 (you can test API locally over SSH).

To inspect failures:

```bash
sudo docker compose -f docker-compose.yml -f deploy/docker-compose.aws.yml logs --tail=100 api
sudo docker compose -f docker-compose.yml -f deploy/docker-compose.aws.yml logs --tail=100 ui
sudo docker compose -f docker-compose.yml -f deploy/docker-compose.aws.yml logs --tail=100 proxy
```

## 5. Update and rollback

Build/test a new release on a staging machine before replacing the versioned model file. Back up the currently approved release ZIP and monitoring volume separately. Deploy the new ZIP and run `up -d --build` again, verifying health and prediction correctness. If a new release fails, restore the previous release and image/model metadata as a matched set. Changing only the model file is rejected by the fingerprint guard.

## 6. Costs and shutdown

Run `sudo docker compose -f docker-compose.yml -f deploy/docker-compose.aws.yml down` to stop containers (does not delete volumes). **EC2 continues billing unless stopped or terminated**, and attached EBS/public IPv4 may also generate charges. After the interview, use the AWS console to stop or terminate the instance and review storage, public IP and billing resources. Avoid `down -v` unless you explicitly intend to delete monitoring and certificate volumes.

## Scaling explanation for interview

This is a reliable *single-host demonstration configuration*, not autoscaling. For a real service, evolve to an image registry (ECR), ECS/Fargate + load balancer, S3 model artifacts with artifact signing, AWS Secrets Manager, centralized CloudWatch monitoring, and a shared datastore for drift aggregation. Add load tests and per-user authentication before processing private customer records.
