let chartInstance = null;
let backtestChartInstance = null;
let distributionChartInstance = null;

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const ticker = document.getElementById('ticker').value.toUpperCase();
    const timeframe = document.getElementById('timeframe').value;
    const modelType = document.getElementById('model-type').value;
    const period = document.getElementById('period').value; // Added period
    const btn = document.getElementById('predict-btn');
    const resultsSection = document.getElementById('results-section');

    // Reset UI
    resultsSection.classList.add('hidden');
    btn.classList.add('loading');
    btn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                ticker: ticker,
                days: parseInt(timeframe),
                days: parseInt(timeframe),
                model_type: modelType,
                period: period, // Added period to body
            })
        });

        if (!response.ok) {
            throw new Error('Prediction failed');
        }

        const data = await response.json();
        renderResults(data);
        resultsSection.classList.remove('hidden');

    } catch (error) {
        console.error('Error:', error);
        alert('Failed to fetch prediction. Please check the ticker symbol and try again.');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
});

document.getElementById('backtest-btn').addEventListener('click', async () => {
    const ticker = document.getElementById('ticker').value.toUpperCase();
    const modelType = document.getElementById('model-type').value;
    const period = document.getElementById('period').value; // Added period
    const btn = document.getElementById('backtest-btn');
    const backtestSection = document.getElementById('backtest-section');

    if (!ticker) {
        alert('Please enter a ticker symbol.');
        return;
    }

    // Reset UI
    backtestSection.classList.add('hidden');
    btn.classList.add('loading');
    btn.disabled = true;
    // const originalText = btn.querySelector('.btn-text').textContent; // No longer needed
    // btn.querySelector('.btn-text').textContent = 'Running...'; // Removed per user request

    // Timeout safeguard
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout

    try {
        const response = await fetch('/backtest', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                ticker: ticker,
                model_type: modelType,
                period: period // Added period to body
            }),
            signal: controller.signal
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server Error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();

        if (!data.results || data.results.length === 0) {
            throw new Error('No backtest results returned from server.');
        }

        renderBacktestResults(data);
        backtestSection.classList.remove('hidden');

    } catch (error) {
        console.error('Error:', error);
        if (error.name === 'AbortError') {
            alert('Backtest timed out. Try a shorter time period or fewer models.');
        } else {
            alert(`Failed to run backtest: ${error.message}`);
        }
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
});

document.getElementById('simulate-btn').addEventListener('click', async () => {
    const ticker = document.getElementById('ticker').value.toUpperCase();
    const timeframe = document.getElementById('timeframe').value;
    const period = document.getElementById('period').value;
    const simulationMethod = document.getElementById('simulation-method').value;
    const btn = document.getElementById('simulate-btn');

    if (!ticker) {
        alert('Please enter a ticker symbol first.');
        return;
    }

    btn.classList.add('loading');
    btn.disabled = true;

    try {
        const response = await fetch('/simulate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                ticker: ticker,
                days: parseInt(timeframe),
                period: period,
                model_type: "monte_carlo",
                simulation_method: simulationMethod
            })
        });

        if (!response.ok) throw new Error('Simulation failed');

        const data = await response.json();
        renderSimulationResults(data, simulationMethod);

    } catch (error) {
        console.error('Error:', error);
        alert('Failed to run simulation.');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
        // Text no longer changes, so no need to revert
    }
});

function renderResults(data) {
    // Update Metrics (Use the first model's metrics or average?)
    // For simplicity, show the first model's metrics or hide if multiple
    document.getElementById('current-price').textContent = formatCurrency(data.current_price);

    if (data.results.length === 1) {
        document.getElementById('model-loss').textContent = data.results[0].metrics.loss.toFixed(5);

        const startPrice = data.results[0].predictions[0].price;
        const endPrice = data.results[0].predictions[data.results[0].predictions.length - 1].price;
        const trend = ((endPrice - startPrice) / startPrice) * 100;
        const trendEl = document.getElementById('prediction-trend');
        trendEl.textContent = `${trend > 0 ? '+' : ''}${trend.toFixed(2)}%`;
        trendEl.style.color = trend >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
    } else {
        document.getElementById('model-loss').textContent = "N/A (Multi)";
        document.getElementById('prediction-trend').textContent = "See Chart";
        document.getElementById('prediction-trend').style.color = 'var(--text-primary)';
    }

    // Render Chart
    initializeChart(data.historical, data.results);

    // Hide Distribution Chart for standard predictions
    document.getElementById('distribution-title').style.display = 'none';
    document.getElementById('distributionChart').parentElement.style.display = 'none';
}

function initializeChart(historicalDataRaw, results = []) {
    const ctx = document.getElementById('predictionChart').getContext('2d');

    if (chartInstance) {
        chartInstance.destroy();
    }

    // Prepare Chart Data
    const historicalData = historicalDataRaw.map(d => ({ x: d.date, y: d.price }));

    const datasets = [
        {
            label: 'Historical Data',
            data: historicalData,
            borderColor: '#9ca3af',
            backgroundColor: 'rgba(156, 163, 175, 0.1)',
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            tension: 0.1
        }
    ];

    // Colors for different models
    const modelColors = {
        'lstm': '#6366f1', // Indigo
        'linear': '#10b981', // Green
        'random_forest': '#f59e0b', // Amber
        'svr': '#ec4899', // Pink
        'gradient_boosting': '#3b82f6', // Blue
        'monte_carlo': '#facc15' // Yellow
    };

    const modelLabels = {
        'lstm': 'LSTM',
        'linear': 'Linear Reg',
        'random_forest': 'Random Forest',
        'svr': 'SVR',
        'gradient_boosting': 'Grad Boosting',
        'monte_carlo': 'Monte Carlo'
    };

    results.forEach(result => {
        const predictionData = result.predictions.map(d => ({ x: d.date, y: d.price }));

        // Connect the last historical point to the first prediction point visually
        const lastHistorical = historicalData[historicalData.length - 1];
        const predictionLineData = [lastHistorical, ...predictionData];

        datasets.push({
            label: modelLabels[result.model] || result.model,
            data: predictionLineData,
            borderColor: modelColors[result.model] || '#ffffff',
            backgroundColor: 'transparent', // Don't fill for multiple lines
            borderWidth: 2,
            pointRadius: 0,
            borderDash: [5, 5],
            fill: false,
            tension: 0.4
        });
    });

    chartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#9ca3af',
                        font: {
                            family: "'Outfit', sans-serif"
                        }
                    }
                },
                tooltip: {
                    backgroundColor: '#181b21',
                    titleColor: '#fff',
                    bodyColor: '#9ca3af',
                    borderColor: '#2d3139',
                    borderWidth: 1,
                    padding: 10,
                    displayColors: true,
                    callbacks: {
                        label: function (context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
                            }
                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: {
                        unit: 'day',
                        displayFormats: {
                            day: 'MMM d'
                        }
                    },
                    grid: {
                        color: '#2d3139'
                    },
                    ticks: {
                        color: '#9ca3af',
                        font: {
                            family: "'Outfit', sans-serif"
                        }
                    }
                },
                y: {
                    grid: {
                        color: '#2d3139'
                    },
                    ticks: {
                        color: '#9ca3af',
                        font: {
                            family: "'Outfit', sans-serif"
                        },
                        callback: function (value) {
                            return '$' + value;
                        }
                    }
                }
            }
        }
    });
}

function renderBacktestResults(data) {
    // Render Metrics
    const metricsContainer = document.getElementById('backtest-metrics');
    metricsContainer.innerHTML = '';

    data.results.forEach(result => {
        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML = `
            <h3>${result.model.toUpperCase()} RMSE</h3>
            <p class="metric-value">${result.metrics.rmse.toFixed(2)}</p>
        `;
        metricsContainer.appendChild(card);
    });

    // Render Chart
    const ctx = document.getElementById('backtestChart').getContext('2d');

    if (backtestChartInstance) {
        backtestChartInstance.destroy();
    }

    // We'll plot Actual vs Predicted for the first result (or all if multiple)
    // If multiple, we plot one Actual line and multiple Predicted lines

    const datasets = [];

    // Add Actual Data (from the first result, as it's the same for all)
    const actualData = data.results[0].dates.map((date, i) => ({ x: date, y: data.results[0].actual[i] }));

    datasets.push({
        label: 'Actual Price',
        data: actualData,
        borderColor: '#9ca3af',
        backgroundColor: 'rgba(156, 163, 175, 0.1)',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.1
    });

    const modelColors = {
        'lstm': '#6366f1',
        'linear': '#10b981',
        'random_forest': '#f59e0b',
        'svr': '#ec4899',
        'gradient_boosting': '#3b82f6',
        'monte_carlo': '#06b6d4'
    };

    data.results.forEach(result => {
        const predictedData = result.dates.map((date, i) => ({ x: date, y: result.predicted[i] }));

        datasets.push({
            label: `${result.model} Prediction`,
            data: predictedData,
            borderColor: modelColors[result.model] || '#ffffff',
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 0,
            borderDash: [2, 2],
            fill: false,
            tension: 0.4
        });
    });

    backtestChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: {
                    labels: { color: '#9ca3af' }
                },
                title: {
                    display: true,
                    text: 'Backtest: Actual vs Predicted (Test Set)',
                    color: '#fff'
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'day' },
                    grid: { color: '#2d3139' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    grid: { color: '#2d3139' },
                    ticks: { color: '#9ca3af' }
                }
            }
        }
    });
}

function renderSimulationResults(data, methodName) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.remove('hidden');

    // Show Distribution Chart for simulations
    document.getElementById('distribution-title').style.display = 'block';
    document.getElementById('distributionChart').parentElement.style.display = 'block';

    // Update Metrics
    document.getElementById('current-price').textContent = formatCurrency(data.current_price);

    // Initialize chart if needed
    if (!chartInstance) {
        initializeChart(data.historical);
    } else {
        // If chart exists, we might want to ensure historical data is up to date or just add to it
        // For simplicity, let's re-init to be safe and clean
        initializeChart(data.historical);
    }

    const dates = data.dates;
    const paths = data.paths;
    const meanPath = data.mean_path;

    // Map method to readable name
    const methodNames = {
        'gbm': 'GBM',
        'jump_diffusion': 'Jump-Diffusion',
        'heston': 'Heston',
        'bootstrapping': 'Bootstrap',
        'ou': 'Ornstein-Uhlenbeck',
        'cev': 'CEV Model',
        'variance_gamma': 'Variance Gamma',
        'cir': 'CIR',
        'arima': 'ARIMA',
        'garch': 'GARCH',
        'lstm_mc': 'LSTM (MC)',
        'vasicek': 'Vasicek'
    };
    const methodName = methodNames[simulationMethod] || 'Monte Carlo';

    // Update Distribution Title
    document.getElementById('distribution-title').textContent = `Price Distribution (${methodName})`;

    // Optimize rendering for 10,000 paths
    // Create a single dataset with nulls to break lines
    const flattenedData = [];

    // We can limit to fewer paths if 10k is still too heavy, but let's try all
    // To avoid browser hang, maybe we sample if it's huge? 
    // But user asked for 10000. Let's try.

    // Optimize rendering: Downsample paths for visualization
    // Rendering 10,000 paths kills mobile performance. We only need ~100 to show the "cloud".
    const maxPathsToRender = 100;
    const step = Math.ceil(paths.length / maxPathsToRender);

    for (let i = 0; i < paths.length; i += step) {
        const path = paths[i];
        path.forEach((price, j) => {
            flattenedData.push({ x: dates[j], y: price });
        });
        // Add a break point to disconnect from the next path
        // We use the last date for the null point to keep x-axis consistent, though it doesn't matter much for null
        flattenedData.push({ x: dates[dates.length - 1], y: null });
    }

    chartInstance.data.datasets.push({
        label: `${methodName} Cloud`,
        data: flattenedData,
        borderColor: 'rgba(255, 255, 255, 0.1)', // Bright White with higher opacity
        backgroundColor: 'transparent',
        borderWidth: 1,
        pointRadius: 0,
        fill: false,
        tension: 0.4,
        order: 10, // Render behind
        spanGaps: false // Important: do NOT span gaps, so nulls break the line
    });

    // Add Mean Path
    const meanData = dates.map((date, i) => ({ x: date, y: meanPath[i] }));
    chartInstance.data.datasets.push({
        label: `${methodName} Mean`,
        data: meanData,
        borderColor: '#facc15', // Solid Yellow
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        borderDash: [5, 5],
        fill: false,
        tension: 0.4,
        order: 1
    });

    chartInstance.update();

    // Render Distribution Chart
    renderDistributionChart(data.distribution);
}

function renderDistributionChart(distribution) {
    const ctx = document.getElementById('distributionChart').getContext('2d');

    if (distributionChartInstance) {
        distributionChartInstance.destroy();
    }

    // Format bin labels (ranges)
    const labels = distribution.bins.map((bin, i) => {
        // Calculate next bin edge (approximate)
        // We can assume uniform width or just show the start
        // Let's try to show range if possible, or just start price
        return formatCurrency(bin);
    });

    distributionChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Frequency',
                data: distribution.counts,
                backgroundColor: 'rgba(250, 204, 21, 0.5)', // Yellow with opacity
                borderColor: '#facc15',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#181b21',
                    titleColor: '#fff',
                    bodyColor: '#9ca3af',
                    callbacks: {
                        title: function (context) {
                            return 'Price: ' + context[0].label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: '#2d3139' },
                    ticks: { color: '#9ca3af', font: { family: "'Outfit', sans-serif" } }
                },
                y: {
                    grid: { color: '#2d3139' },
                    ticks: { color: '#9ca3af', font: { family: "'Outfit', sans-serif" } }
                }
            }
        }
    });
}

// Force Service Worker Unregistration to clear cache
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.getRegistrations().then(function (registrations) {
        for (let registration of registrations) {
            registration.unregister().then(function (boolean) {
                console.log('Service Worker unregistered: ', boolean);
            });
        }
    });
}

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}
