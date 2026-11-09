# Deploying on AWS

How the API runs on AWS, every command in order, and how to update, stop and fix it. First deployed 2026-11-09.
`<ACCOUNT_ID>` is the 12-digit AWS account number (not a secret, but kept out of this public repository).
What to do when an alert goes off: [RUNBOOK.md](RUNBOOK.md).

## How it fits together

```
Laptop (Mac, Arm)                         AWS, Ohio (us-east-2)
-----------------                         ----------------------------------------------------------
docker build  ──docker push──►  ECR repository "cart-rag"  (stores the image, tag v1, v2, ...)
                                              │ docker pull
                                              ▼
                                EC2 computer "cart-rag-server"  (t4g.small, Arm, 2 GB memory, 20 GB disk)
                                  Docker container "cart-rag"
                                    ├─ API key  ◄── Parameter Store "/cart-rag/ANTHROPIC_API_KEY" (encrypted)
                                    ├─ logbook, cache, usage counts  → Docker volume "cart-rag-state" (on the disk)
                                    └─ port 80 (HTTP) ◄── browsers
                                  IAM role "cart-rag-server": what the computer itself is allowed to do
```

| AWS part | What it does here |
|---|---|
| **ECR** | Stores the Docker image. The computer downloads it from here. |
| **EC2** | The rented computer that runs the container. |
| **EBS** | The computer's disk (20 GB, gp3). Keeps the image and the logbook volume. |
| **Security group** | The firewall: only HTTP (80) and HTTPS (443) are open. SSH (22) is closed. |
| **Session Manager** | Opens a terminal on the computer in the browser (EC2 → Connect). No key file, no open SSH port. |
| **Parameter Store** | Keeps the Claude API key encrypted (SecureString). Never in the code, the image or GitHub. |
| **IAM role** | The computer may only download images (`AmazonEC2ContainerRegistryReadOnly`) and use Session Manager and read parameters (`AmazonSSMManagedInstanceCore`). |
| **Budgets** | Emails when spending nears $10 a month. |

## One-time setup (already done)

**Account:** MFA on the root login; a $10 monthly cost budget. The account is on the Free plan ($100 credits),
which pins it to us-east-2 and blocks IAM users with console passwords and IAM Identity Center, so the root login is
used with MFA and never gets access keys. Least privilege is applied to the server's role instead.

**Laptop:** install the AWS CLI (version 2.32 or newer, from `https://awscli.amazonaws.com/AWSCLIV2.pkg`), then

```bash
aws login --region us-east-2        # signs in through the browser (with MFA); a temporary pass, up to 12 hours
aws configure set region us-east-2
```

Log in again with the same command when it says the session has expired.

**Image storage (ECR):**

```bash
aws ecr create-repository --repository-name cart-rag --region us-east-2
```

**API key (AWS website):** Parameter Store → Create parameter: name `/cart-rag/ANTHROPIC_API_KEY`, tier Standard,
type **SecureString**, default key `alias/aws/ssm`, value = a Claude key made only for the server (Anthropic
Console, named `aws-server`), so it can be deleted without affecting other keys.

**Role (AWS website):** IAM → Roles → Create role → AWS service, EC2 → attach `AmazonSSMManagedInstanceCore` and
`AmazonEC2ContainerRegistryReadOnly` → name `cart-rag-server`.

**Computer (AWS website):** EC2 → Launch instance: name `cart-rag-server`; Amazon Linux, **64-bit (Arm)**;
**t4g.small**; no key pair; network: SSH off, HTTP on (HTTPS on, for later); storage **20 GiB gp3**; Advanced
details → IAM instance profile `cart-rag-server`.

- Arm because the image is built on an Arm Mac. Building on an Intel/AMD machine needs
  `docker build --platform linux/arm64 -t cart-rag .`
- t4g.small because the app uses ~1 GB at rest and ~1.4 GB under load (`eval/load_test.py`); t4g.micro has 1 GB.
- No file system (EFS): SQLite should not be on a network drive.

**On the computer** (EC2 → Instances → `cart-rag-server` → Connect → Session Manager → Connect):

```bash
sudo dnf install -y docker
sudo systemctl enable --now docker          # start Docker now and after every restart
```

## Putting a version online

**1. On the laptop: build and upload.** Use a new tag each time (`v2`, `v3`, ...), so the previous version stays
available to go back to.

```bash
python src/slim_index.py                    # only if the index changed
docker build -t cart-rag .
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com
docker tag cart-rag:latest <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/cart-rag:v2
docker push <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/cart-rag:v2
```

Test the image locally first (README, Docker): `/health` should say `"status":"ok"`.

**2. On the computer (Session Manager): download it and swap the running container.**

```bash
aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com
sudo docker pull <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/cart-rag:v2
sudo docker rm -f cart-rag                  # stop and remove the old container (the logbook volume is kept)
sudo docker run -d --name cart-rag --restart unless-stopped -p 80:8000 -v cart-rag-state:/app/state \
  -e ANTHROPIC_API_KEY="$(aws ssm get-parameter --name /cart-rag/ANTHROPIC_API_KEY --with-decryption --query Parameter.Value --output text --region us-east-2)" \
  <ACCOUNT_ID>.dkr.ecr.us-east-2.amazonaws.com/cart-rag:v2
sudo docker ps                              # after ~1 minute: "Up ... (healthy)"
```

| Part of `docker run` | Meaning |
|---|---|
| `-d --name cart-rag` | Run in the background, named `cart-rag` |
| `--restart unless-stopped` | Start again by itself after a crash or a reboot |
| `-p 80:8000` | The website port (80) goes to the app inside (8000) |
| `-v cart-rag-state:/app/state` | Logbook, cache and usage counts survive new versions |
| `-e ANTHROPIC_API_KEY="$(aws ssm ...)"` | Reads the key from Parameter Store at start; it is never printed |

**3. Check:** `http://<public IP>/health` in a browser (EC2 → Instances → `cart-rag-server` → Public IPv4
address). Type `http://` in full: browsers may try HTTPS on their own, which is not set up yet, and time out.

**Going back:** run step 2 with the previous tag (e.g. `v1`).

**Changing the API key:** update the parameter in Parameter Store, then run the `docker rm -f` and `docker run`
lines again (the key is read only when the container starts).

## Stopping and starting

- **Stop** when not needed: EC2 → Instances → `cart-rag-server` → Instance state → Stop. The computer's charge
  stops; the disk (~$1.60/month) is still charged. Start it the same way; Docker and the app start by themselves.
- **The public IP changes** after every stop and start.
- **Terminate** deletes the computer and its disk, including the logbook. Not needed for pausing.

## Costs (Ohio, from the $100 credits)

| Item | About |
|---|---|
| t4g.small, running all the time | $12/month |
| 20 GB gp3 disk | $1.60/month |
| Public IPv4 address | $3.60/month |
| ECR storage (one or two versions) | a few cents/month |
| Parameter Store (Standard), Session Manager, the role | free |
| Claude API (paid to Anthropic, not AWS) | ~2.5 cents per new answer; capped at $1/day by `api/usage.py` |

## When something is wrong

| What you see | Check |
|---|---|
| Browser: "took too long to respond" | Type `http://` in full. Then: is the computer Running? Security group → Inbound rules has port 80 from `0.0.0.0/0`? |
| Page does not load, computer running | Session Manager: `sudo docker ps`. Not listed: `sudo docker ps -a` and `sudo docker logs cart-rag --tail 50` |
| `/health` shows `API key ... not set` | The `aws ssm get-parameter` part failed: check the parameter name and that the role is attached (EC2 → instance → Security → IAM role) |
| `docker pull`: "no basic auth credentials" | Run the `aws ecr get-login-password ... docker login` line again (it lasts 12 hours) |
| Laptop: "session expired" / "token" errors | `aws login --region us-east-2` |
| `exec format error` in the logs | The image was built for the wrong chip: rebuild with `--platform linux/arm64` |
| Alerts at `/health` | [RUNBOOK.md](RUNBOOK.md) |

Logs and the logbook: `sudo docker logs cart-rag --tail 50`; the report runs inside the container:
`sudo docker exec cart-rag python api/log_report.py --days 1`.

## Not done yet

- **HTTPS.** Until then, do not share the link: questions travel unencrypted.
- **A fixed address** (Elastic IP or a domain), so the address does not change on restart.
- **Monitoring from outside:** CloudWatch alarms and an uptime check on `/health`.
- **Load test on the server** (`eval/load_test.py` was run on a laptop).
- The key is passed to the container as an environment variable, so anyone with admin rights on the computer can
  see it (`docker inspect`). Acceptable with one admin; reading it inside the app would avoid that.
