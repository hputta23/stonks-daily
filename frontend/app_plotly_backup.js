// Global State for Chart Re-rendering

// State variables (global scope so functions can access)
let currentHistoricalData = [];
let currentResults = [];


function renderResults(data) {
    // Store global state
    currentHistoricalData = data.historical;
    currentResults = data.results;

    document.getElementById('current-price').textContent = formatCurrency(data.current_price);

    if (data.results.length === 1) {
        const metrics = data.results[0].metrics || {};
        const loss = metrics.loss !== undefined ? metrics.loss.toFixed(5) : 'N/A';
        document.getElementById('model-loss').textContent = loss;

        const startPrice = data.results[0].predictions[0].price;
        const endPrice = data.results[0].predictions[data.results[0].predictions.length - 1].price;
        const trend = ((endPrice - startPrice) / startPrice) * 100;
        const trendEl = document.getElementById('prediction-trend');
        trendEl.textContent = `${trend > 0 ? '+' : ''}${trend.toFixed(2)}% `;
        trendEl.style.color = trend >= 0 ? 'var(--accent-green)' : 'var(--accent-red)';
    } else {
        document.getElementById('model-loss').textContent = "N/A (Multi)";
        document.getElementById('prediction-trend').textContent = "See Chart";
        document.getElementById('prediction-trend').style.color = 'var(--text-primary)';
    }

    renderInteractiveChart(data.historical, data.results);

    // Hide Distribution Chart for standard predictions
    document.getElementById('distribution-title').style.display = 'none';
    document.getElementById('distributionChart').parentElement.style.display = 'none';
}

function renderInteractiveChart(historical, results = []) {
    const colors = getThemeColors();
    const chartType = document.getElementById('chart-type').value;
    const showVolume = document.getElementById('show-volume').checked;
    const showSMA = document.getElementById('show-sma').checked;
    const showEMA = document.getElementById('show-ema').checked;
    const showBollinger = document.getElementById('show-bollinger').checked;
    const showRSI = document.getElementById('show-rsi').checked;
    const showMACD = document.getElementById('show-macd').checked;

    const data = [];
    const layout = {
        plot_bgcolor: '#191c24',
        paper_bgcolor: '#191c24',
        font: { color: '#9ca3af', family: "'Outfit', sans-serif" },
        dragmode: 'pan',
        showlegend: true,
        hovermode: 'x unified',
        xaxis: {
            type: 'date',
            gridcolor: '#2d3139',
            rangeslider: { visible: false },
            rangeselector: {
                buttons: [
                    { count: 1, label: '1m', step: 'month', stepmode: 'backward' },
                    { count: 6, label: '6m', step: 'month', stepmode: 'backward' },
                    { count: 1, label: 'YTD', step: 'year', stepmode: 'todate' },
                    { count: 1, label: '1y', step: 'year', stepmode: 'backward' },
                    { step: 'all' }
                ],
                bgcolor: '#374151',
                activecolor: '#4f46e5',
                font: { color: '#ffffff' }
            }
        },
        yaxis: {
            title: 'Price ($)',
            gridcolor: '#2d3139',
            fixedrange: false // Allow vertical zoom
        },
        // Grid layout definition (subplots)
        grid: { rows: 1, columns: 1, pattern: 'independent' },
        margin: { l: 40, r: 10, t: 20, b: 40, pad: 0 },
        autosize: true
    };

    // Calculate Grid Layout
    let rows = 1;
    let rowHeights = [1];
    if (showRSI) rows++;
    if (showMACD) rows++;

    // Adjust y-axis domains based on number of subplots
    // 0 is bottom, 1 is top. Main chart at top.
    // Logic: Stack them. Main chart takes huge chunk, indicators small chunks at bottom.
    // Plotly domains: [start, end].

    // Simplification: We will assign axis numbers (y, y2, y3...)
    // Main chart: y (always).
    // Volume: overlay on main chart or separate? User wants volume.
    // Usually Volume is subplot bottom of price or overlay. Let's do subplot 2 (overlaying y doesn't scale well).
    // Actually, widespread practice is separate pane or overlay with separate axis.
    // Let's optimize: Volume = bottom 20% of Main Chart (overlay y-axis, short bars).

    // Let's implement RSI and MACD as separate rows below.
    // Standard approach: 
    // Row 1 (Main): Top 60-70%
    // Row 2 (RSI): 15%
    // Row 3 (MACD): 15%

    let mainDomainEnd = 1.0;
    let currentY = 0;

    // Base Traces (Price)
    const dates = historical.map(d => d.date);
    let priceTrace;

    if (chartType === 'candlestick') {
        priceTrace = {
            x: dates,
            close: historical.map(d => d.close),
            high: historical.map(d => d.high),
            low: historical.map(d => d.low),
            open: historical.map(d => d.open),
            decreasing: { line: { color: '#ef4444' } },
            increasing: { line: { color: '#10b981' } },
            line: { color: 'rgba(31,119,180,1)' },
            type: 'candlestick',
            xaxis: 'x',
            yaxis: 'y',
            name: 'OHLC'
        };
    } else {
        priceTrace = {
            x: dates,
            y: historical.map(d => d.close),
            type: 'scatter',
            mode: 'lines',
            line: { color: '#9ca3af', width: 2 },
            name: 'Close Price'
        };
    }
    data.push(priceTrace);

    // Indicators (Overlay)
    if (showSMA) {
        data.push({ x: dates, y: historical.map(d => d.sma_20), type: 'scatter', mode: 'lines', name: 'SMA 20', line: { color: '#fb923c', width: 1 } }); // Orange
        data.push({ x: dates, y: historical.map(d => d.sma_50), type: 'scatter', mode: 'lines', name: 'SMA 50', line: { color: '#f59e0b', width: 1 } }); // Amber
    }
    if (showEMA) {
        data.push({ x: dates, y: historical.map(d => d.ema_12), type: 'scatter', mode: 'lines', name: 'EMA 12', line: { color: '#38bdf8', width: 1 } }); // Sky
    }
    if (showBollinger) {
        data.push({ x: dates, y: historical.map(d => d.upper_band), type: 'scatter', mode: 'lines', name: 'Upper BB', line: { color: 'rgba(255, 255, 255, 0.3)', width: 0 }, showlegend: false });
        data.push({ x: dates, y: historical.map(d => d.lower_band), type: 'scatter', mode: 'lines', name: 'Bollinger', fill: 'tonexty', fillcolor: 'rgba(255, 255, 255, 0.05)', line: { color: 'rgba(255, 255, 255, 0.3)', width: 0 } });
    }

    // Predictions
    const modelColors = { 'lstm': '#6366f1', 'random_forest': '#f59e0b', 'gradient_boosting': '#3b82f6' };
    results.forEach(result => {
        const predDates = result.predictions.map(d => d.date);
        const predPrices = result.predictions.map(d => d.price);

        // Connect lines
        const lastHistDate = dates[dates.length - 1];
        const lastHistPrice = historical[historical.length - 1].close;

        const plotDates = [lastHistDate, ...predDates];
        const plotPrices = [lastHistPrice, ...predPrices];

        let traceColor = modelColors[result.model] || '#ff00ff';
        let traceWidth = 2;
        let traceOpacity = 1.0;
        let showLegend = true;
        let hoverInfo = 'y+name';

        if (result.isSimulation) {
            traceColor = colors.textSecondary; // or a nice subtle grey/blue
            traceWidth = 1;
            traceOpacity = 0.05; // Very transparent
            showLegend = false;
            hoverInfo = 'skip';
        }

        data.push({
            x: plotDates,
            y: plotPrices,
            type: 'scatter',
            mode: 'lines',
            name: result.isSimulation ? '' : result.model.toUpperCase() + ' Pred',
            line: { color: traceColor, dash: result.isSimulation ? 'solid' : 'dash', width: traceWidth },
            opacity: traceOpacity,
            showlegend: showLegend,
            hoverinfo: hoverInfo
        });
    });

    // Subplots Logic
    // Plotly requires layout.yaxis, layout.yaxis2 etc.
    // 'y' is main.

    // If showVolume, we map it to y-axis but create a separate axis overlay or subplot?
    // Let's try putting Volume on a y-axis2 that overlays y but is restricted to bottom 20%?
    // Or better: distinct rows.

    // Let's define domains dynamically.
    const axisConfigs = {};
    let currentTop = 1.0;
    const gap = 0.05;

    // Calculate total slots
    let slots = 1; // Main
    if (showRSI) slots += 0.3;
    if (showMACD) slots += 0.3;
    if (showVolume) slots += 0.2; // Volume smaller

    // Normalize to 1.0
    // Actually simpler: 
    // Main Panel height = remaining space.
    // Bottom panels fixed height approx 0.15 or 0.2.

    let bottomPointer = 0;
    let layouts = [];

    if (showMACD) {
        layouts.push({ name: 'MACD', height: 0.15, id: 'macd' });
    }
    if (showRSI) {
        layouts.push({ name: 'RSI', height: 0.15, id: 'rsi' });
    }
    if (showVolume) {
        layouts.push({ name: 'Vol', height: 0.15, id: 'vol' });
    }

    // Assign domains
    // Bottom up
    let yIndex = 2; // y2, y3...

    layouts.forEach(l => {
        const domainStart = bottomPointer;
        const domainEnd = bottomPointer + l.height;
        bottomPointer += (l.height + gap);

        const yAxisName = 'yaxis' + ((yIndex > 1) ? yIndex : '');
        layout[yAxisName] = {
            title: l.name,
            domain: [domainStart, domainEnd],
            gridcolor: colors.grid,
            fixedrange: false
        };

        // Add Trace
        if (l.id === 'vol') {
            data.push({
                x: dates,
                y: historical.map(d => d.volume),
                type: 'bar',
                name: 'Volume',
                marker: { color: '#374151' },
                yaxis: 'y' + yIndex
            });
        }
        else if (l.id === 'rsi') {
            data.push({
                x: dates,
                y: historical.map(d => d.rsi),
                type: 'scatter',
                mode: 'lines',
                name: 'RSI',
                line: { color: '#a855f7' },
                yaxis: 'y' + yIndex
            });
            // Reference lines 30/70
            layout[yAxisName].range = [0, 100];
            // We could add shapes for 30/70 but simple lines ok
        }
        else if (l.id === 'macd') {
            data.push({
                x: dates,
                y: historical.map(d => d.macd),
                type: 'bar', // Histogram
                name: 'MACD Hist',
                marker: { color: '#ec4899' },
                yaxis: 'y' + yIndex
            });
            data.push({
                x: dates,
                y: historical.map(d => d.signal_line),
                type: 'scatter',
                mode: 'lines',
                name: 'Signal',
                line: { color: '#facc15' },
                yaxis: 'y' + yIndex
            });
        }

        yIndex++;
    });

    // Main Axis (always 'y')
    layout.yaxis.domain = [bottomPointer, 1.0];
    // Link x-axes? Implicit in Plotly shares x usually if not specified differently, but best to anchor.
    // Actually we just use 'x' for all traces.

    // Render
    Plotly.newPlot('predictionChart', data, layout, { responsive: true, displayModeBar: true });
}

function renderBacktestResults(data) {
    const metricsContainer = document.getElementById('backtest-metrics');
    metricsContainer.innerHTML = '';

    data.results.forEach(result => {
        const metrics = result.metrics;
        const confidenceMsg = metrics.confidence_interval
            ? `< p class="metric-sub" >± ${formatCurrency(metrics.confidence_interval)} (95 % CI)</p > `
            : '';

        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML = `
    < h3 > ${result.model.toUpperCase()} Performance</h3 >
            <div class="metric-row">
                <div>
                    <span class="metric-label">RMSE</span>
                    <p class="metric-value">${metrics.rmse.toFixed(2)}</p>
                </div>
                <div>
                    <span class="metric-label">MAE</span>
                    <p class="metric-value">${metrics.mae.toFixed(2)}</p>
                </div>
            </div>
            <div class="metric-row">
                <div>
                    <span class="metric-label">R² Score</span>
                    <p class="metric-value">${metrics.r2 ? metrics.r2.toFixed(3) : 'N/A'}</p>
                </div>
                 <div>
                    <span class="metric-label">MAPE</span>
                    <p class="metric-value">${metrics.mape ? (metrics.mape * 100).toFixed(1) + '%' : 'N/A'}</p>
                </div>
            </div>
            
            <h4 style="margin-top: 1rem; color: #9ca3af; font-size: 0.9em; text-transform: uppercase;">Risk Analysis (Forecast)</h4>
            <div class="metric-row">
                <div>
                   <span class="metric-label">Sharpe Ratio</span>
                   <p class="metric-value" style="color: ${metrics.sharpe >= 1 ? '#10b981' : (metrics.sharpe > 0 ? '#fb923c' : '#ef4444')}">${metrics.sharpe ? metrics.sharpe.toFixed(2) : '0.00'}</p>
                </div>
                <div>
                   <span class="metric-label">Volatility (Ann.)</span>
                   <p class="metric-value">${metrics.volatility ? (metrics.volatility * 100).toFixed(1) + '%' : '0.0%'}</p>
                </div>
            </div>
            <div class="metric-row">
                <div>
                   <span class="metric-label">VaR (95%)</span>
                   <p class="metric-value" style="color: #ef4444">${metrics.var_95 ? (metrics.var_95 * 100).toFixed(2) + '%' : '0.00%'}</p>
                </div>
                <div>
                   <span class="metric-label">Max Drawdown</span>
                   <p class="metric-value" style="color: #ef4444">${metrics.max_drawdown ? (metrics.max_drawdown * 100).toFixed(2) + '%' : '0.00%'}</p>
                </div>
            </div>
            
            ${confidenceMsg}
`;
        metricsContainer.appendChild(card);

        if (result.feature_importance && Object.keys(result.feature_importance).length > 0) {
            document.getElementById('feature-importance-section').classList.remove('hidden');
            renderFeatureImportanceChart(result.feature_importance);
        } else {
            document.getElementById('feature-importance-section').classList.add('hidden');
        }
    });

    // Plotly Backtest Chart
    const result = data.results[0];
    const colors = getThemeColors();

    const traceActual = {
        x: result.dates,
        y: result.actual,
        type: 'scatter',
        mode: 'lines',
        name: 'Actual',
        line: { color: colors.textSecondary, width: 2 } // Adapted for light mode visibility
    };

    const tracePred = {
        x: result.dates,
        y: result.predicted,
        type: 'scatter',
        mode: 'lines',
        name: 'Predicted',
        line: { color: '#6366f1', width: 2, dash: 'dash' }
    };

    const layout = {
        title: { text: 'Backtest: Actual vs Predicted', font: { color: colors.text } },
        plot_bgcolor: colors.bg,
        paper_bgcolor: colors.bg,
        font: { color: colors.textSecondary, family: "'Outfit', sans-serif" },
        showlegend: true,
        hovermode: 'x unified',
        xaxis: { gridcolor: colors.grid },
        yaxis: { gridcolor: colors.grid, title: 'Price ($)' },
        margin: { l: 40, r: 10, t: 40, b: 40, pad: 0 },
        autosize: true
    };

    Plotly.newPlot('backtestChart', [traceActual, tracePred], layout, { responsive: true, displayModeBar: true });
}

function renderFeatureImportanceChart(featureImportance) {
    const labels = Object.keys(featureImportance);
    const values = Object.values(featureImportance);
    const colors = getThemeColors();

    // Sort logic handled in backend usually, but ensures plot direction

    const data = [{
        type: 'bar',
        x: values,
        y: labels,
        orientation: 'h',
        marker: {
            color: 'rgba(99, 102, 241, 0.8)',
            line: { color: '#6366f1', width: 1 }
        }
    }];

    const layout = {
        plot_bgcolor: colors.bg,
        paper_bgcolor: colors.bg,
        font: { color: colors.textSecondary, family: "'Outfit', sans-serif" },
        margin: { l: 150, r: 20, t: 10, b: 60 }, // Left margin for labels
        xaxis: { gridcolor: colors.grid, title: 'Importance Score' },
        yaxis: { gridcolor: 'transparent' }
    };

    Plotly.newPlot('featureImportanceChart', data, layout, { responsive: true, displayModeBar: true });
}

function renderSimulationResults(data, simulationMethod) {
    const resultsSection = document.getElementById('results-section');
    resultsSection.classList.remove('hidden');

    // Update Metrics
    document.getElementById('current-price').textContent = formatCurrency(data.current_price);

    // Setup Global Data simulating 'Results' format for interactive chart
    // Simulation logic is slightly different, usually just lines.
    // We can use the same renderInteractiveChart if we massage data.

    currentHistoricalData = data.historical;
    currentResults = [];

    // 1. Add Sample Paths (Cloud) - Limit to 200 to prevent crash
    if (data.paths && data.paths.length > 0) {
        const sampleSize = 200;
        const samples = data.paths.slice(0, sampleSize);

        samples.forEach((path, idx) => {
            currentResults.push({
                model: `Sim ${idx}`,
                isSimulation: true,
                predictions: path.map((price, i) => ({ date: data.dates[i], price: price }))
            });
        });
    }

    // 2. Add Mean Path
    const simResult = {
        model: 'Mean Path',
        predictions: data.mean_path.map((price, i) => ({ date: data.dates[i], price: price }))
    };
    currentResults.push(simResult);

    renderInteractiveChart(currentHistoricalData, currentResults);

    // Add the "Cloud" if we want? The renderInteractiveChart is generic. 
    // To add the cloud trace, we would need to manually extend the Plotly data 
    // AFTER calling renderInteractiveChart, or make standard function smarter.
    // For now, let's just show Mean path in standard chart, and Distribution below.

    document.getElementById('distribution-title').style.display = 'block';
    document.getElementById('distributionChart').parentElement.style.display = 'block';

    renderDistributionChart(data.distribution);
}

function renderDistributionChart(distribution) {
    // Plotly Distribution
    // Midpoints of bins for x-axis? Or proper histogram.
    // distribution.bins are edges.
    const colors = getThemeColors();

    const data = [{
        x: distribution.bins,
        y: distribution.counts,
        type: 'bar',
        name: 'Freq',
        marker: { color: '#facc15' }
    }];

    const layout = {
        title: { text: 'Price Distribution (End)', font: { color: colors.text } },
        plot_bgcolor: colors.bg,
        paper_bgcolor: colors.bg,
        font: { color: colors.textSecondary, family: "'Outfit', sans-serif" },
        xaxis: { gridcolor: colors.grid, title: 'Price' },
        yaxis: { gridcolor: colors.grid, title: 'Frequency' },
        margin: { l: 40, r: 10, t: 40, b: 40, pad: 0 },
        autosize: true
    };

    Plotly.newPlot('distributionChart', data, layout, { responsive: true, displayModeBar: true });
}

// Utils
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.getRegistrations().then(function (registrations) {
        for (let registration of registrations) {
            registration.unregister();
        }
    });
}

function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

// --- Theme Handling & Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    console.log("App.js Initialized - DOMContentLoaded");

    // Theme Logic
    const themeToggle = document.getElementById('theme-toggle');
    const html = document.documentElement;
    const savedTheme = localStorage.getItem('theme') || 'dark';
    if (savedTheme === 'light') {
        html.setAttribute('data-theme', 'light');
        if (themeToggle) themeToggle.textContent = '☀️';
    }

    if (themeToggle) {
        themeToggle.addEventListener('click', () => {
            const currentTheme = html.getAttribute('data-theme');
            if (currentTheme === 'light') {
                html.removeAttribute('data-theme');
                localStorage.setItem('theme', 'dark');
                themeToggle.textContent = '🌙';
            } else {
                html.setAttribute('data-theme', 'light');
                localStorage.setItem('theme', 'light');
                themeToggle.textContent = '☀️';
            }
            updateChartsTheme();
        });
    }

    function updateChartsTheme() {
        const isLight = html.getAttribute('data-theme') === 'light';
        const bgColor = isLight ? '#ffffff' : '#181b21';
        const textColor = isLight ? '#111827' : '#ffffff';
        const gridColor = isLight ? '#e5e7eb' : '#2d3139';

        const update = {
            'plot_bgcolor': bgColor,
            'paper_bgcolor': bgColor,
            'font.color': textColor,
            'xaxis.gridcolor': gridColor,
            'yaxis.gridcolor': gridColor
        };

        const charts = ['predictionChart', 'distributionChart', 'backtestChart', 'featureImportanceChart'];
        charts.forEach(id => {
            const el = document.getElementById(id);
            try {
                if (el && el.innerHTML !== "") {
                    Plotly.relayout(id, update).catch(err => console.log("Chart not ready", err));
                }
            } catch (e) { }
        });
    }

    // --- Event Listeners (Moved Inside DOMContentLoaded) ---

    // 1. Prediction Form
    const predictForm = document.getElementById('prediction-form');
    if (predictForm) {
        predictForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            console.log("Prediction Form Submitted");

            const ticker = document.getElementById('ticker').value.toUpperCase();
            const timeframe = document.getElementById('timeframe').value;
            const modelType = document.getElementById('model-type').value;
            const period = document.getElementById('period').value;
            const btn = document.getElementById('predict-btn');
            const resultsSection = document.getElementById('results-section');

            resultsSection.classList.add('hidden');
            btn.classList.add('loading');
            btn.disabled = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        ticker: ticker,
                        days: parseInt(timeframe),
                        model_type: modelType,
                        period: period,
                    })
                });

                if (!response.ok) {
                    const txt = await response.text();
                    throw new Error('Prediction failed: ' + txt);
                }

                const data = await response.json();
                renderResults(data);
                resultsSection.classList.remove('hidden');

            } catch (error) {
                console.error('Error:', error);
                alert('Failed to fetch prediction: ' + error.message);
            } finally {
                btn.classList.remove('loading');
                btn.disabled = false;
            }
        });
    }

    // 2. Backtest Form
    const backtestForm = document.getElementById('backtest-form');
    if (backtestForm) {
        backtestForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const ticker = document.getElementById('ticker').value.toUpperCase();
            const period = document.getElementById('period').value;
            const btn = document.getElementById('backtest-btn');
            const backtestSection = document.getElementById('backtest-results');

            if (!ticker) { alert('Please enter a ticker symbol.'); return; }

            backtestSection.classList.add('hidden');
            btn.classList.add('loading');
            btn.disabled = true;

            try {
                const response = await fetch('/backtest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ticker: ticker, period: period })
                });

                if (!response.ok) throw new Error('Backtest failed');
                const data = await response.json();
                renderBacktestResults(data);
                backtestSection.classList.remove('hidden');
                // Scroll to backtest results
                backtestSection.scrollIntoView({ behavior: 'smooth' });

            } catch (error) {
                console.error('Error:', error);
                alert('Backtest failed: ' + error.message);
            } finally {
                btn.classList.remove('loading');
                btn.disabled = false;
            }
        });
    }

    // 3. Simulation Button
    const simulateBtn = document.getElementById('simulate-btn');
    if (simulateBtn) {
        simulateBtn.addEventListener('click', async () => {
            console.log("Simulate Clicked");
            const ticker = document.getElementById('ticker').value.toUpperCase();
            const timeframe = document.getElementById('timeframe').value;
            const period = document.getElementById('period').value;
            const simulationMethod = document.getElementById('simulation-method').value;

            if (!ticker) {
                alert('Please enter a ticker symbol first.');
                return;
            }

            simulateBtn.classList.add('loading');
            simulateBtn.disabled = true;

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
                simulateBtn.classList.remove('loading');
                simulateBtn.disabled = false;
            }
        });
    }

    // 4. Chart Controls
    const chartControls = ['chart-type', 'show-volume', 'show-sma', 'show-ema', 'show-bollinger', 'show-rsi', 'show-macd'];
    chartControls.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', () => {
                if (typeof currentHistoricalData !== 'undefined' && currentHistoricalData.length > 0) {
                    renderInteractiveChart(currentHistoricalData, currentResults);
                }
            });
        }
    });

    // Update Diagnostics
    const jsStatus = document.getElementById('js-status');
    if (jsStatus) {
        jsStatus.innerText = 'OK (Ready)';
        jsStatus.style.color = '#0f0';
    }
}); // Close DOMContentLoaded

function getThemeColors() {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    return {
        bg: isLight ? '#ffffff' : '#181b21',
        text: isLight ? '#111827' : '#ffffff',
        grid: isLight ? '#e5e7eb' : '#2d3139',
        border: isLight ? '#e5e7eb' : '#2d3139',
        textSecondary: isLight ? '#4b5563' : '#9ca3af'
    };
}
