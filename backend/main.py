from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from .data_service import fetch_stock_data, get_current_price
from .data_service import fetch_stock_data, get_current_price
from .model import get_predictor
import traceback
import os

app = FastAPI()

# Mount static files
# Get absolute path to frontend directory
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, "../frontend")

# Mount frontend directory at /static to serve assets
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/manifest.json")
async def get_manifest():
    return FileResponse('frontend/manifest.json', media_type='application/json')

@app.get("/sw.js")
async def get_sw():
    return FileResponse('frontend/sw.js', media_type='application/javascript')

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    ticker: str
    days: int = 30
    model_type: str = "lstm"
    period: str = "2y"
    run_simulation: bool = False
    simulation_method: str = "gbm"



@app.get("/")
async def read_root():
    return FileResponse(os.path.join(frontend_dir, 'index.html'))

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # 1. Fetch Data
        data = fetch_stock_data(request.ticker, period=request.period)
        
        # 2. Train Models to run
        models_to_run = []
        if request.model_type == "all":
            models_to_run = ["lstm", "random_forest", "svr", "gradient_boosting", "monte_carlo"]
        else:
            models_to_run = [request.model_type]
            
        if request.run_simulation:
            if "monte_carlo" not in models_to_run:
                models_to_run.append("monte_carlo")
            
        results = []
        
        for model_name in models_to_run:
            # Train Model
            predictor = get_predictor(model_name)
            # Train on all available data
            history, scaled_data = predictor.train(data, epochs=20) 
            
            # Predict Future
            future_dates, future_prices = predictor.predict_future(data, days=request.days)
            
            predictions = []
            for date, price in zip(future_dates, future_prices):
                predictions.append({
                    "date": date.isoformat(),
                    "price": float(price)
                })
                
            results.append({
                "model": model_name,
                "predictions": predictions,
                "metrics": {
                    "loss": float(history.history['loss'][-1]) if history and 'loss' in history.history else 0
                }
            })
        
        # 4. Prepare Response
        # Convert timestamps to ISO strings
        historical_data = []
        for index, row in data.iterrows():
            historical_data.append({
                "date": row['Date'].isoformat(),
                "price": float(row['Close'])
            })
            
        current_price = get_current_price(request.ticker)
        
        return {
            "ticker": request.ticker,
            "current_price": current_price,
            "historical": historical_data[-45:], # Return last 45 days (Zoomed In)
            "results": results
        }
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate")
async def simulate(request: PredictionRequest):
    try:
        # 1. Fetch Data
        data = fetch_stock_data(request.ticker, period=request.period)
        
        # 2. Get Monte Carlo Predictor
        predictor = get_predictor("monte_carlo")
        predictor.train(data) # Calculate drift/volatility
        
        # 3. Predict Paths
        # Generate 10000 paths for visualization (optimized rendering on frontend)
        future_dates, mean_path, paths = predictor.predict_paths(data, days=request.days, iterations=10000, method=request.simulation_method)
        
        # Format dates
        dates = [d.isoformat() for d in future_dates]
        
        # Prepare historical data for chart
        historical_data = []
        for index, row in data.iterrows():
            historical_data.append({
                "date": row['Date'].isoformat(),
                "price": float(row['Close'])
            })
            
        current_price = get_current_price(request.ticker)
        
        # Calculate Distribution
        final_prices = [path[-1] for path in paths]
        hist, bin_edges = np.histogram(final_prices, bins=20)
        
        distribution = {
            "bins": [float(b) for b in bin_edges[:-1]], # Start of each bin
            "counts": [int(c) for c in hist]
        }

        return {
            "ticker": request.ticker,
            "current_price": current_price,
            "historical": historical_data[-45:], # Return last 45 days (Zoomed In)
            "dates": dates,
            "mean_path": [float(p) for p in mean_path],
            "paths": [[float(p) for p in path] for path in paths],
            "distribution": distribution
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/backtest")
async def backtest(request: PredictionRequest):
    try:
        # 1. Fetch Data (fetch more data for backtesting, e.g., 2 years)
        data = fetch_stock_data(request.ticker, period="2y")
        
        # 2. Get Model
        # If "all" is selected, maybe just backtest LSTM for now or loop?
        # Let's support single model backtest for simplicity first, or loop if requested.
        # The UI will likely pass a specific model or "all".
        
        models_to_test = []
        if request.model_type == "all":
            models_to_test = ["lstm", "random_forest", "svr", "gradient_boosting", "monte_carlo"]
        else:
            models_to_test = [request.model_type]
            
        results = []
        
        for model_name in models_to_test:
            predictor = get_predictor(model_name)
            backtest_result = predictor.backtest(data)
            
            # Format dates
            dates = [pd.to_datetime(d).isoformat() for d in backtest_result['dates']]
            
            results.append({
                "model": model_name,
                "dates": dates,
                "actual": [float(x) for x in backtest_result['actual']],
                "predicted": [float(x) for x in backtest_result['predicted']],
                "metrics": backtest_result['metrics']
            })
            
        return {
            "ticker": request.ticker,
            "results": results
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
