// TradingView Lightweight Charts Implementation
// Global chart instances
let mainChart = null;
let candlestickSeries = null;
let volumeSeries = null;
let predictionLines = [];
let simulationLines = [];

// Current results storage
let currentResults = [];
let currentHistorical = [];

// Theme colors helper
function getThemeColors() {
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    return {
        bg: isDark ? '#191c24' : '#ffffff',
        text: isDark ? '#ffffff' : '#000000',
        textSecondary: isDark ? '#9ca3af' : '#666666',
        grid: isDark ? '#2d3139' : '#e5e7eb',
        volumeUp: isDark ? '#10b981' : '#16a34a',
        volumeDown: isDark ? '#ef4444' : '#dc2626'
    };
}

// Initialize main chart
function createMainChart() {
    const container = document.getElementById('predictionChart');
    container.innerHTML = ''; // Clear any existing content

    const colors = getThemeColors();

    mainChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            background: { color: colors.bg },
            textColor: colors.textSecondary,
        },
        grid: {
            vertLines: { color: colors.grid },
            horzLines: { color: colors.grid },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: colors.grid,
        },
        timeScale: {
            borderColor: colors.grid,
            timeVisible: true,
            secondsVisible: false,
        },
    });

    // Create candlestick series
    candlestickSeries = mainChart.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
    });

    // Create volume series (histogram overlay)
    volumeSeries = mainChart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: 'volume',
        scaleMargins: {
            top: 0.8,
            bottom: 0,
        },
    });

    // Handle window resize
    new ResizeObserver(entries => {
        if (mainChart) {
            mainChart.applyOptions({
                width: entries[0].contentRect.width
            });
        }
    }).observe(container);
}

// Convert data to TradingView format
function convertToTVFormat(historical) {
    const candleData = historical.map(d => ({
        time: new Date(d.date).getTime() / 1000, // Convert to Unix timestamp
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close
    }));

    const volumeData = historical.map(d => ({
        time: new Date(d.date).getTime() / 1000,
        value: d.volume,
        color: d.close >= d.open ? '#10b981' : '#ef4444'
    }));

    return { candleData, volumeData };
}

// Render prediction results
function renderPredictionResults(data) {
    console.log('[TradingView] Rendering prediction results:', data);

    // Store historical data
    currentHistorical = data.historical;

    // Create or update chart
    if (!mainChart) {
        createMainChart();
    }

    // Convert and set historical data
    const { candleData, volumeData } = convertToTVFormat(data.historical);
    candlestickSeries.setData(candleData);
    volumeSeries.setData(volumeData);

    // Clear previous prediction lines
    predictionLines.forEach(series => mainChart.removeSeries(series));
    predictionLines = [];

    // Store results for later overlays
    currentResults = data.results || [];

    // Add prediction lines
    data.results.forEach((result, index) => {
        if (!result.predictions || result.predictions.length === 0) return;

        // Get color for model
        const modelColors = {
            'random_forest': '#f59e0b',
            'lstm': '#3b82f6',
            'ensemble': '#a855f7'
        };
        const color = modelColors[result.model] || '#6366f1';

        // Create line series for prediction
        const lineSeries = mainChart.addLineSeries({
            color: color,
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            title: result.model.toUpperCase() + ' Prediction',
        });

        // Convert predictions to TV format
        const predData = result.predictions.map((pred, i) => ({
            time: new Date(result.prediction_dates[i]).getTime() / 1000,
            value: pred
        }));

        // Connect last historical point to first prediction
        const lastHistorical = data.historical[data.historical.length - 1];
        const connectionPoint = {
            time: new Date(lastHistorical.date).getTime() / 1000,
            value: lastHistorical.close
        };

        lineSeries.setData([connectionPoint, ...predData]);
        predictionLines.push(lineSeries);
    });

    // Fit content
    mainChart.timeScale().fitContent();

    // Update metrics
    updateMetrics(data);

    // Show results section
    document.getElementById('results-section').classList.remove('hidden');
}

// Update metrics display
function updateMetrics(data) {
    const metricsContainer = document.getElementById('prediction-metrics');
    metricsContainer.innerHTML = '';

    data.results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML = `
            <h3>${result.model.replace('_', ' ').toUpperCase()}</h3>
            <div class="metric-value">$${result.last_prediction.toFixed(2)}</div>
        `;
        metricsContainer.appendChild(card);
    });
}

// Render simulation cloud
function renderSimulationResults(data) {
    console.log('[TradingView] Rendering simulation cloud');

    // Clear previous simulation lines
    simulationLines.forEach(series => mainChart.removeSeries(series));
    simulationLines = [];

    if (!data.paths || data.paths.length === 0) return;

    // Sample 200 paths from 10,000
    const sampleSize = Math.min(200, data.paths.length);
    const step = Math.floor(data.paths.length / sampleSize);

    for (let i = 0; i < data.paths.length; i += step) {
        if (simulationLines.length >= sampleSize) break;

        const path = data.paths[i];

        // Create very transparent line series
        const lineSeries = mainChart.addLineSeries({
            color: 'rgba(156, 163, 175, 0.05)',
            lineWidth: 1,
            lastValueVisible: false,
            priceLineVisible: false,
        });

        // Convert to TV format
        const pathData = path.map((value, idx) => ({
            time: new Date(data.prediction_dates[idx]).getTime() / 1000,
            value: value
        }));

        // Connect to last historical point
        const lastHistorical = currentHistorical[currentHistorical.length - 1];
        const connectionPoint = {
            time: new Date(lastHistorical.date).getTime() / 1000,
            value: lastHistorical.close
        };

        lineSeries.setData([connectionPoint, ...pathData]);
        simulationLines.push(lineSeries);
    }

    // Add mean path on top
    const meanSeries = mainChart.addLineSeries({
        color: '#facc15',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        title: 'Simulation Mean',
    });

    const meanData = data.mean_path.map((value, idx) => ({
        time: new Date(data.prediction_dates[idx]).getTime() / 1000,
        value: value
    }));

    const lastHistorical = currentHistorical[currentHistorical.length - 1];
    meanSeries.setData([
        {
            time: new Date(lastHistorical.date).getTime() / 1000,
            value: lastHistorical.close
        },
        ...meanData
    ]);

    predictionLines.push(meanSeries);
}

// Render backtest results
function renderBacktestResults(data) {
    const colors = getThemeColors();
    const container = document.getElementById('backtestChart');
    container.innerHTML = '';

    const backtestChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight,
        layout: {
            background: { color: colors.bg },
            textColor: colors.textSecondary,
        },
        grid: {
            vertLines: { color: colors.grid },
            horzLines: { color: colors.grid },
        },
    });

    data.results.forEach(result => {
        // Actual line
        const actualSeries = backtestChart.addLineSeries({
            color: '#10b981',
            lineWidth: 2,
            title: 'Actual',
        });

        const actualData = result.actual.map((value, idx) => ({
            time: new Date(result.dates[idx]).getTime() / 1000,
            value: value
        }));
        actualSeries.setData(actualData);

        // Predicted line
        const predSeries = backtestChart.addLineSeries({
            color: '#6366f1',
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            title: result.model.toUpperCase() + ' Pred',
        });

        const predData = result.predicted.map((value, idx) => ({
            time: new Date(result.dates[idx]).getTime() / 1000,
            value: value
        }));
        predSeries.setData(predData);
    });

    backtestChart.timeScale().fitContent();

    // Handle resize
    new ResizeObserver(entries => {
        backtestChart.applyOptions({ width: entries[0].contentRect.width });
    }).observe(container);

    // Update metrics
    const metricsContainer = document.getElementById('backtest-metrics');
    metricsContainer.innerHTML = '';

    data.results.forEach(result => {
        const metrics = result.metrics;
        ['mse', 'rmse', 'mae', 'sharpe_ratio', 'max_drawdown', 'var_95'].forEach(key => {
            if (metrics[key] !== undefined) {
                const card = document.createElement('div');
                card.className = 'metric-card';
                card.innerHTML = `
                    <h3>${key.replace('_', ' ').toUpperCase()}</h3>
                    <div class="metric-value">${metrics[key].toFixed(4)}</div>
                `;
                metricsContainer.appendChild(card);
            }
        });
    });

    document.getElementById('backtest-section').classList.remove('hidden');
}

// Event listeners setup
document.addEventListener('DOMContentLoaded', () => {
    console.log('[TradingView] App initialized');

    // Prediction form
    const predictionForm = document.getElementById('prediction-form');
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const ticker = document.getElementById('ticker').value;
        const days = document.getElementById('days').value;
        const model = document.getElementById('model').value;

        const predictBtn = document.querySelector('#predict-btn');
        predictBtn.disabled = true;
        predictBtn.textContent = 'Predicting...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker, days: parseInt(days), model })
            });

            const data = await response.json();
            renderPredictionResults(data);
        } catch (error) {
            console.error('Prediction error:', error);
            alert('Prediction failed: ' + error.message);
        } finally {
            predictBtn.disabled = false;
            predictBtn.textContent = 'Generate Prediction';
        }
    });

    // Simulate button
    const simulateBtn = document.getElementById('simulate-btn');
    simulateBtn.addEventListener('click', async () => {
        const ticker = document.getElementById('ticker').value;
        const days = document.getElementById('days').value;

        simulateBtn.disabled = true;
        simulateBtn.textContent = 'Simulating...';

        try {
            const response = await fetch('/simulate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker, days: parseInt(days) })
            });

            const data = await response.json();
            renderSimulationResults(data);
        } catch (error) {
            console.error('Simulation error:', error);
        } finally {
            simulateBtn.disabled = false;
            simulateBtn.textContent = 'Simulate';
        }
    });

    // Backtest button
    const backtestBtn = document.getElementById('backtest-btn');
    backtestBtn.addEventListener('click', async () => {
        const ticker = document.getElementById('ticker').value;
        const model = document.getElementById('model').value;

        backtestBtn.disabled = true;
        backtestBtn.textContent = 'Running...';

        try {
            const response = await fetch('/backtest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker, model })
            });

            const data = await response.json();
            renderBacktestResults(data);
        } catch (error) {
            console.error('Backtest error:', error);
        } finally {
            backtestBtn.disabled = false;
            backtestBtn.textContent = 'Run Backtest';
        }
    });

    // Theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = document.documentElement.getAttribute('data-theme');
            const newTheme = currentTheme === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);

            // Recreate charts with new theme
            if (currentHistorical.length > 0 && currentResults.length > 0) {
                createMainChart();
                const data = {
                    historical: currentHistorical,
                    results: currentResults
                };
                renderPredictionResults(data);
            }
        });
    }
});
