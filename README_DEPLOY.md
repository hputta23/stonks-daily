# 🚀 Deployment Guide: Stonks Daily

You can host this application for **free** on platforms like **Render**, **Railway**, or **Heroku**. This guide focuses on **Render** as it's the easiest for this type of app.

## Option 1: Deploy to Render (Easiest & Free)

1.  **Push to GitHub**:
    *   Create a new repository on GitHub.
    *   Push this entire `stock_prediction` folder to it.

2.  **Create Service on Render**:
    *   Go to [dashboard.render.com](https://dashboard.render.com/).
    *   Click **"New +"** -> **"Web Service"**.
    *   Connect your GitHub repository.

3.  **Configure**:
    *   **Runtime**: Python 3
    *   **Build Command**: `pip install -r requirements.txt` (Render should auto-detect `runtime.txt` for Python 3.9.18)
    *   **Start Command**: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
    *   (Or just let it auto-detect the `Procfile` I created for you).
    *   **Instance Type**: Free

4.  **Deploy**:
    *   Click "Create Web Service".
    *   Wait ~5 minutes. Render will give you a URL like `https://stonks-daily.onrender.com`.

## Option 2: Run with Docker

If you have Docker Desktop installed, you can build and run it locally in a container:

```bash
# Build the image
docker build -t stonks-daily .

# Run the container (mapping port 8000)
docker run -p 8000:8000 -e PORT=8000 stonks-daily
```

## files Created for You
*   `Dockerfile`: Instructions for Docker to build your app.
*   `Procfile`: Instructions for Heroku/Railway.
*   `requirements.txt`: List of all Python libraries needed.
