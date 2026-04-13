# Deploying to Azure App Service (no Docker)

## Resources required

| Resource | Tier | Cost |
|---|---|---|
| App Service Plan | B1 Basic | ~$13/month |
| Web App (Python 3.11) | — | Included in plan |

> **Why B1 not F1 (free)?**  Streamlit uses WebSockets.  
> Azure App Service only enables WebSockets on Basic tier (B1) and above.  
> F1 shows a blank/disconnected screen.
>
> You can **stop the app** between sessions to pause billing.

---

## Before you deploy

1. Install [Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli)
2. Edit **config.yaml** — set `app.data_dir` to `/home/data`:

```yaml
app:
  data_dir: "/home/data"   # persistent mount on Azure App Service
```

> `/home/data` is an Azure Files mount built into App Service — data survives restarts.  
> `config.yaml` is excluded from git via `.gitignore`; it gets deployed as part of the zip.

---

## Deploy with one command

Run from inside the `Leaderboard` folder:

```bash
az login
az webapp up \
  --name sports-leaderboard \
  --resource-group <your-resource-group> \
  --runtime "PYTHON:3.11" \
  --sku B1 \
  --location eastus
```

> Pick a unique `--name` — it becomes `https://sports-leaderboard.azurewebsites.net`.

---

## Set the startup command

```bash
az webapp config set \
  --name sports-leaderboard \
  --resource-group <your-resource-group> \
  --startup-file "python run.py"
```

---

## Enable WebSockets

```bash
az webapp config set \
  --name sports-leaderboard \
  --resource-group <your-resource-group> \
  --web-sockets-enabled true
```

---

## Set the exposed port

```bash
az webapp config appsettings set \
  --name sports-leaderboard \
  --resource-group <your-resource-group> \
  --settings WEBSITES_PORT="8501"
```

---

## Restart and get your URL

```bash
az webapp restart \
  --name sports-leaderboard \
  --resource-group <your-resource-group>

az webapp show \
  --name sports-leaderboard \
  --resource-group <your-resource-group> \
  --query defaultHostName --output tsv
```

Share the URL with all players — they open it on any phone browser.

---

## Redeploy after code changes

```bash
az webapp up \
  --name sports-leaderboard \
  --resource-group <your-resource-group>
```

---

## Stop billing between sessions

```bash
# Stop (pauses billing)
az webapp stop --name sports-leaderboard --resource-group <your-resource-group>

# Start again next time
az webapp start --name sports-leaderboard --resource-group <your-resource-group>
```

---

## Tear down everything

```bash
az group delete --name <your-resource-group> --yes --no-wait
```

---

## What's inside your resource group

```
Your Resource Group
├── sports-leaderboard-plan   (App Service Plan  B1)
└── sports-leaderboard        (Web App  Python 3.11)
```

No Container Registry, no database, no storage account — nothing else needed.
