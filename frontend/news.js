class StockNewsApp {
    constructor() {
        console.log('StockNewsApp constructor called');
        // Initialize elements lazily or check if they exist
        this.elements = {};
        this.sentiment = new FinancialSentiment();
        this.cachedNews = {};

        // Settings
        this.recentTickers = JSON.parse(localStorage.getItem('stonks_news_recent')) || ['AAPL', 'TSLA', 'NVDA'];

        // We will call init manually or when tab is active
    }

    init() {
        if (this.initialized) return;

        this.cacheElements();
        this.bindEvents();
        this.loadDashboard();

        this.initialized = true;
    }

    cacheElements() {
        this.elements = {
            searchSection: document.getElementById('news-search-section'),
            tickerInput: document.getElementById('news-ticker-input'),
            searchBtn: document.getElementById('news-search-btn'),
            homeBtn: document.getElementById('news-home-btn'),
            newsContainer: document.getElementById('news-container'),
            searchResultsContainer: document.getElementById('news-search-results-container'),
            searchResults: document.getElementById('news-search-results'),
            marketOverviewHeader: document.getElementById('market-overview-header'),
            loadingIndicator: document.getElementById('news-loading'),
            errorMessage: document.getElementById('news-error-message'),
            errorText: document.getElementById('news-error-text')
        };
    }

    bindEvents() {
        if (this.elements.searchBtn) {
            this.elements.searchBtn.addEventListener('click', () => this.handleSearch());
        }
        if (this.elements.tickerInput) {
            this.elements.tickerInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.handleSearch();
            });
        }
        if (this.elements.homeBtn) {
            this.elements.homeBtn.addEventListener('click', () => this.resetDefaults());
        }
    }

    resetDefaults() {
        if (confirm('Reset news search history?')) {
            localStorage.removeItem('stonks_news_recent');
            this.recentTickers = ['AAPL', 'TSLA', 'NVDA'];
            this.loadDashboard();
        }
    }

    async loadDashboard() {
        // Dashboard Mode
        if (this.elements.tickerInput) this.elements.tickerInput.value = '';
        if (this.elements.searchResultsContainer) this.elements.searchResultsContainer.classList.add('hidden');
        if (this.elements.marketOverviewHeader) this.elements.marketOverviewHeader.classList.remove('hidden');
        if (this.elements.newsContainer) this.elements.newsContainer.innerHTML = '';

        this.setError(null);
        this.setLoading(true);

        try {
            const uniqueTickers = [...new Set(this.recentTickers)].slice(0, 3);
            const promises = uniqueTickers.map(ticker => this.fetchAndRenderTickerSection(ticker, this.elements.newsContainer));
            await Promise.allSettled(promises);
        } catch (error) {
            console.error('Dashboard load error', error);
        } finally {
            this.setLoading(false);
        }
    }

    async handleSearch() {
        const ticker = this.elements.tickerInput.value.trim().toUpperCase();
        if (!ticker) {
            this.loadDashboard();
            return;
        }

        // Search Mode
        if (this.elements.searchResultsContainer) this.elements.searchResultsContainer.classList.remove('hidden');
        if (this.elements.searchResults) this.elements.searchResults.innerHTML = '';

        this.setLoading(true);
        this.setError(null);

        try {
            // 1. Fetch main search
            const success = await this.fetchAndRenderTickerSection(ticker, this.elements.searchResults, true);

            if (success) {
                this.updateRecentTickers(ticker);
            }

            // 2. Refresh Market Overview
            if (this.elements.newsContainer) this.elements.newsContainer.innerHTML = '';
            const uniqueTickers = [...new Set(this.recentTickers)].slice(0, 3);
            const promises = uniqueTickers.map(t => this.fetchAndRenderTickerSection(t, this.elements.newsContainer));
            await Promise.allSettled(promises);

        } catch (error) {
            this.setError(error.message);
        } finally {
            this.setLoading(false);
        }
    }

    updateRecentTickers(ticker) {
        this.recentTickers.unshift(ticker);
        this.recentTickers = [...new Set(this.recentTickers)].slice(0, 3);
        localStorage.setItem('stonks_news_recent', JSON.stringify(this.recentTickers));
    }

    async fetchAndRenderTickerSection(ticker, container, isSingleView = false) {
        if (!container) return false;

        try {
            const news = await this.fetchNews(ticker);
            this.cachedNews[ticker] = { items: news, sort: 'desc' };

            const section = this.createTickerSection(ticker, news, isSingleView);
            container.appendChild(section);
            return true;
        } catch (error) {
            console.error(`Failed to load ${ticker}`, error);
            if (isSingleView) {
                this.setError(`Could not fetch news for ${ticker}.`);
            } else {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'ticker-section error-card';
                errorDiv.innerHTML = `<h3><span class="badge">${ticker}</span></h3><p class="error-text">Failed to load news.</p>`;
                container.appendChild(errorDiv);
            }
            return false;
        }
    }

    createTickerSection(ticker, newsItems, isSingleView) {
        const section = document.createElement('div');
        section.className = isSingleView ? 'single-view' : 'ticker-section';

        const header = document.createElement('h3');
        header.innerHTML = isSingleView
            ? `<span class="badge">${ticker}</span> Latest News`
            : `<span class="badge">${ticker}</span>`;
        section.appendChild(header);

        const tableContainer = document.createElement('div');
        tableContainer.className = 'table-container';

        console.log(`Rendering news for ${ticker}. Items:`, newsItems);

        if (!newsItems || newsItems.length === 0) {
            tableContainer.innerHTML = '<div class="empty-state"><p>No news found.</p></div>';
            section.appendChild(tableContainer);
            return section;
        }

        const limit = isSingleView ? 20 : 5;
        const displayItems = newsItems.slice(0, limit);

        const table = document.createElement('table');
        table.className = 'news-table';

        const uniqueId = `tbody-${ticker}-${Math.random().toString(36).substr(2, 9)}`;

        table.innerHTML = `
            <thead>
                <tr>
                    <th style="width: 60%;">Headline</th>
                    <th style="width: 20%;">Source</th>
                    <th class="date-header sortable" data-ticker="${ticker}" data-tbody="${uniqueId}" style="width: 20%;">Date ↓</th>
                </tr>
            </thead>
            <tbody id="${uniqueId}"></tbody>
        `;

        // Event listener for sort
        table.querySelector('.date-header').addEventListener('click', (e) => {
            this.toggleSort(ticker, e.currentTarget);
        });

        const tbody = table.querySelector('tbody');
        this.renderRows(tbody, displayItems);

        tableContainer.appendChild(table);
        section.appendChild(tableContainer);
        return section;
    }

    renderRows(tbody, items) {
        tbody.innerHTML = '';
        items.forEach(item => {
            const row = document.createElement('tr');

            const date = new Date(item.datetime * 1000).toLocaleString('en-US', {
                month: 'short', day: 'numeric'
            });

            const analysis = this.sentiment.analyze(item.headline);
            const sentimentClass = analysis.type === 'positive' ? 'sentiment-positive' :
                analysis.type === 'negative' ? 'sentiment-negative' : '';

            row.innerHTML = `
                <td><a href="${item.url}" target="_blank" class="news-headline headline-link ${sentimentClass}" title="${item.headline}">${item.headline}</a></td>
                <td class="news-source">${item.source}</td>
                <td>${date}</td>
            `;
            tbody.appendChild(row);
        });
    }

    toggleSort(ticker, headerElement) {
        const data = this.cachedNews[ticker];
        if (!data) return;

        data.sort = data.sort === 'desc' ? 'asc' : 'desc';
        data.items.sort((a, b) => data.sort === 'desc' ? b.datetime - a.datetime : a.datetime - b.datetime);

        // Find associated tbody
        const tbodyId = headerElement.dataset.tbody;
        const tbody = document.getElementById(tbodyId);
        if (!tbody) return;

        // Determine limit based on context (hacky check for parent class)
        const isSingleView = tbody.closest('.single-view') !== null;
        const limit = isSingleView ? 20 : 5;

        this.renderRows(tbody, data.items.slice(0, limit));
        headerElement.textContent = `Date ${data.sort === 'desc' ? '↓' : '↑'}`;
    }

    async fetchNews(ticker) {
        try {
            const response = await fetch(`/news/${ticker}`);
            if (!response.ok) throw new Error('Failed to fetch news');

            const data = await response.json();
            return data.news;
        } catch (error) {
            console.error('News fetch error:', error);
            throw error;
        }
    }

    setLoading(isLoading) {
        if (this.elements.loadingIndicator) {
            this.elements.loadingIndicator.classList.toggle('hidden', !isLoading);
        }
    }

    setError(msg) {
        if (this.elements.errorMessage) {
            this.elements.errorMessage.classList.toggle('hidden', !msg);
            if (this.elements.errorText) this.elements.errorText.textContent = msg || '';
        }
    }
}

// Make globally available
window.StockNewsApp = StockNewsApp;
console.log('news.js loaded, StockNewsApp exposed');
