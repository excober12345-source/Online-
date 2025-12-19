// TradeMaster Mobile Trading App
// Main Application Logic

class TradingApp {
    constructor() {
        this.currentSymbol = 'AAPL';
        this.portfolio = {
            balance: 10000,
            holdings: [
                { symbol: 'AAPL', shares: 10, avgPrice: 170.50 },
                { symbol: 'MSFT', shares: 5, avgPrice: 330.25 },
                { symbol: 'GOOGL', shares: 2, avgPrice: 135.75 }
            ]
        };
        this.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'BTC-USD', 'ETH-USD'];
        this.priceData = {};
        this.chart = null;
        
        this.init();
    }
    
    init() {
        // Update time every minute
        this.updateTime();
        setInterval(() => this.updateTime(), 60000);
        
        // Load initial data
        this.loadMarketData();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Simulate loading completion
        setTimeout(() => {
            document.getElementById('loader').style.display = 'none';
            document.getElementById('app').style.display = 'block';
        }, 1500);
        
        // Initialize chart
        this.initChart();
        
        // Load portfolio
        this.updatePortfolioDisplay();
        
        // Load watchlist
        this.updateWatchlist();
        
        // Load news
        this.loadNews();
    }
    
    updateTime() {
        const now = new Date();
        const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        document.getElementById('currentTime').textContent = timeString;
    }
    
    async loadMarketData() {
        try {
            // For demo purposes, using mock data
            // In production, you would fetch from a real API
            
            const mockPrices = {
                'AAPL': { price: 175.25, change: 2.15, changePercent: 1.24, name: 'Apple Inc.' },
                'MSFT': { price: 332.50, change: -1.25, changePercent: -0.37, name: 'Microsoft Corp.' },
                'GOOGL': { price: 138.75, change: 0.85, changePercent: 0.62, name: 'Alphabet Inc.' },
                'AMZN': { price: 145.30, change: 3.20, changePercent: 2.25, name: 'Amazon.com Inc.' },
                'TSLA': { price: 245.80, change: -5.40, changePercent: -2.15, name: 'Tesla Inc.' },
                'NVDA': { price: 485.25, change: 12.75, changePercent: 2.70, name: 'NVIDIA Corp.' },
                'BTC-USD': { price: 42500.50, change: 1250.75, changePercent: 3.03, name: 'Bitcoin' },
                'ETH-USD': { price: 2250.30, change: 45.25, changePercent: 2.05, name: 'Ethereum' }
            };
            
            this.priceData = mockPrices;
            
            // Update current symbol display
            this.updateCurrentSymbol();
            
        } catch (error) {
            console.error('Error loading market data:', error);
        }
    }
    
    updateCurrentSymbol() {
        const symbolData = this.priceData[this.currentSymbol];
        if (!symbolData) return;
        
        document.getElementById('currentSymbol').textContent = this.currentSymbol;
        document.getElementById('companyName').textContent = symbolData.name;
        document.getElementById('currentPrice').textContent = `$${symbolData.price.toFixed(2)}`;
        
        const changeClass = symbolData.change >= 0 ? 'positive' : 'negative';
        const changeSign = symbolData.change >= 0 ? '+' : '';
        document.getElementById('priceChange').textContent = 
            `${changeSign}${symbolData.change.toFixed(2)} (${changeSign}${symbolData.changePercent.toFixed(2)}%)`;
        document.getElementById('priceChange').className = `price-change ${changeClass}`;
        
        // Update trade panel price
        document.getElementById('tradePrice').textContent = `$${symbolData.price.toFixed(2)}`;
        
        // Update max shares
        const maxShares = Math.floor(this.portfolio.balance / symbolData.price);
        document.getElementById('maxShares').textContent = maxShares;
        
        // Update chart
        this.updateChart();
    }
    
    initChart() {
        const ctx = document.getElementById('priceChart').getContext('2d');
        
        // Generate mock chart data
        const labels = [];
        const data = [];
        let currentPrice = this.priceData[this.currentSymbol]?.price || 175;
        
        for (let i = 0; i < 24; i++) {
            labels.push(`${i}:00`);
            // Simulate price movement
            const change = (Math.random() - 0.5) * 2;
            currentPrice += change;
            data.push(currentPrice);
        }
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Price',
                    data: data,
                    borderColor: 'rgba(255, 255, 255, 0.8)',
                    backgroundColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
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
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)'
                        }
                    },
                    y: {
                        display: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)'
                        }
                    }
                }
            }
        });
    }
    
    updateChart() {
        if (!this.chart) return;
        
        // Update chart with new data
        const currentPrice = this.priceData[this.currentSymbol]?.price || 175;
        const newData = [];
        
        for (let i = 0; i < 24; i++) {
            const change = (Math.random() - 0.5) * 2;
            const price = currentPrice + change * (24 - i);
            newData.push(price);
        }
        
        this.chart.data.datasets[0].data = newData;
        this.chart.update();
    }
    
    updatePortfolioDisplay() {
        // Calculate portfolio value
        let totalValue = this.portfolio.balance;
        let todayChange = 0;
        
        this.portfolio.holdings.forEach(holding => {
            const currentPrice = this.priceData[holding.symbol]?.price || 0;
            totalValue += holding.shares * currentPrice;
            todayChange += holding.shares * (this.priceData[holding.symbol]?.change || 0);
        });
        
        const changePercent = (todayChange / (totalValue - todayChange)) * 100;
        
        // Update display
        document.getElementById('portfolioValue').textContent = `$${totalValue.toFixed(2)}`;
        document.getElementById('portfolioChange').textContent = 
            `$${todayChange.toFixed(2)} (${changePercent.toFixed(2)}%) Today`;
        document.getElementById('portfolioChange').className = 
            todayChange >= 0 ? 'positive' : 'negative';
        
        document.getElementById('availableBalance').textContent = `$${this.portfolio.balance.toFixed(2)}`;
        
        // Update holdings list
        const holdingsList = document.getElementById('portfolioHoldings');
        holdingsList.innerHTML = '';
        
        this.portfolio.holdings.forEach(holding => {
            const currentPrice = this.priceData[holding.symbol]?.price || 0;
            const value = holding.shares * currentPrice;
            const pnl = (currentPrice - holding.avgPrice) * holding.shares;
            const pnlPercent = ((currentPrice - holding.avgPrice) / holding.avgPrice) * 100;
            
            const holdingElement = document.createElement('div');
            holdingElement.className = 'watchlist-item';
            holdingElement.innerHTML = `
                <div>
                    <div class="symbol">${holding.symbol}</div>
                    <div class="company">${holding.shares} shares</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 600;">$${value.toFixed(2)}</div>
                    <div class="${pnl >= 0 ? 'positive' : 'negative'}" style="font-size: 14px;">
                        ${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} (${pnl >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%)
                    </div>
                </div>
            `;
            
            holdingsList.appendChild(holdingElement);
        });
    }
    
    updateWatchlist() {
        const watchlistElement = document.getElementById('watchlist');
        watchlistElement.innerHTML = '';
        
        this.watchlist.forEach(symbol => {
            const symbolData = this.priceData[symbol];
            if (!symbolData) return;
            
            const item = document.createElement('div');
            item.className = 'watchlist-item';
            item.style.cursor = 'pointer';
            item.onclick = () => {
                this.currentSymbol = symbol;
                this.updateCurrentSymbol();
                showTab('dashboard');
                
                // Update nav items
                document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
                document.querySelector('.nav-item:nth-child(1)').classList.add('active');
            };
            
            item.innerHTML = `
                <div>
                    <div class="symbol">${symbol}</div>
                    <div class="company">${symbolData.name}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 600;">$${symbolData.price.toFixed(2)}</div>
                    <div class="${symbolData.change >= 0 ? 'positive' : 'negative'}" style="font-size: 14px;">
                        ${symbolData.change >= 0 ? '+' : ''}$${symbolData.change.toFixed(2)} (${symbolData.change >= 0 ? '+' : ''}${symbolData.changePercent.toFixed(2)}%)
                    </div>
                </div>
            `;
            
            watchlistElement.appendChild(item);
        });
    }
    
    async loadNews() {
        // Mock news data
        const news = [
            { title: 'Federal Reserve Holds Rates Steady', source: 'Bloomberg', time: '2 hours ago' },
            { title: 'Apple Unveils New AI Features', source: 'CNBC', time: '4 hours ago' },
            { title: 'Tesla Reports Record Deliveries', source: 'Reuters', time: '6 hours ago' },
            { title: 'Bitcoin Surges Past $42,000', source: 'CoinDesk', time: '8 hours ago' }
        ];
        
        const newsList = document.getElementById('newsList');
        newsList.innerHTML = '';
        
        news.forEach(item => {
            const newsItem = document.createElement('div');
            newsItem.className = 'watchlist-item';
            newsItem.innerHTML = `
                <div>
                    <div style="font-weight: 600; margin-bottom: 4px;">${item.title}</div>
                    <div style="font-size: 14px; color: #6b7280;">
                        ${item.source} · ${item.time}
                    </div>
                </div>
            `;
            newsList.appendChild(newsItem);
        });
    }
    
    setupEventListeners() {
        // Swipe to refresh
        let startY = 0;
        const appContent = document.querySelector('.app-content');
        
        appContent.addEventListener('touchstart', (e) => {
            startY = e.touches[0].clientY;
        });
        
        appContent.addEventListener('touchmove', (e) => {
            const currentY = e.touches[0].clientY;
            const diff = currentY - startY;
            
            if (diff > 50 && window.scrollY === 0) {
                document.getElementById('refreshIndicator').style.display = 'block';
            }
        });
        
        appContent.addEventListener('touchend', (e) => {
            const currentY = e.changedTouches[0].clientY;
            const diff = currentY - startY;
            
            if (diff > 100 && window.scrollY === 0) {
                this.refreshData();
            }
            
            setTimeout(() => {
                document.getElementById('refreshIndicator').style.display = 'none';
            }, 1000);
        });
        
        // Trade panel events
        document.getElementById('tradeQuantity').addEventListener('input', () => {
            this.updateTradeTotal();
        });
        
        document.getElementById('executeTradeBtn').addEventListener('click', () => {
            this.executeTrade();
        });
        
        // Add to Home Screen prompt
        this.checkPWAInstall();
    }
    
    refreshData() {
        // Simulate data refresh
        this.loadMarketData();
        this.updatePortfolioDisplay();
        this.updateWatchlist();
        
        // Show refresh animation
        const indicator = document.getElementById('refreshIndicator');
        indicator.style.display = 'block';
        
        setTimeout(() => {
            indicator.style.display = 'none';
            this.showNotification('Data refreshed successfully!');
        }, 1000);
    }
    
    updateTradeTotal() {
        const quantity = parseInt(document.getElementById('tradeQuantity').value) || 0;
        const price = this.priceData[this.currentSymbol]?.price || 0;
        const total = quantity * price;
        const commission = total * 0.001; // 0.1% commission
        const netAmount = total + commission;
        
        document.getElementById('totalCost').textContent = `$${total.toFixed(2)}`;
        document.getElementById('commission').textContent = `$${commission.toFixed(2)}`;
        document.getElementById('netAmount').textContent = `$${netAmount.toFixed(2)}`;
    }
    
    executeTrade() {
        const tradeType = document.getElementById('tradeType').textContent.split(' ')[0];
        const quantity = parseInt(document.getElementById('tradeQuantity').value) || 0;
        const price = this.priceData[this.currentSymbol]?.price || 0;
        const total = quantity * price;
        const commission = total * 0.001;
        
        if (tradeType === 'BUY') {
            if (total + commission > this.portfolio.balance) {
                this.showNotification('Insufficient funds!', 'error');
                return;
            }
            
            this.portfolio.balance -= (total + commission);
            
            // Add to holdings or update existing
            const existingHolding = this.portfolio.holdings.find(h => h.symbol === this.currentSymbol);
            if (existingHolding) {
                const totalShares = existingHolding.shares + quantity;
                const newAvgPrice = ((existingHolding.shares * existingHolding.avgPrice) + total) / totalShares;
                existingHolding.shares = totalShares;
                existingHolding.avgPrice = newAvgPrice;
            } else {
                this.portfolio.holdings.push({
                    symbol: this.currentSymbol,
                    shares: quantity,
                    avgPrice: price
                });
            }
            
            this.showNotification(`Bought ${quantity} ${this.currentSymbol} for $${total.toFixed(2)}`);
            
        } else if (tradeType === 'SELL') {
            const existingHolding = this.portfolio.holdings.find(h => h.symbol === this.currentSymbol);
            
            if (!existingHolding || existingHolding.shares < quantity) {
                this.showNotification('Insufficient shares to sell!', 'error');
                return;
            }
            
            this.portfolio.balance += (total - commission);
            existingHolding.shares -= quantity;
            
            if (existingHolding.shares === 0) {
                this.portfolio.holdings = this.portfolio.holdings.filter(h => h.symbol !== this.currentSymbol);
            }
            
            this.showNotification(`Sold ${quantity} ${this.currentSymbol} for $${total.toFixed(2)}`);
        }
        
        // Update displays
        this.updatePortfolioDisplay();
        this.closeTradePanel();
        
        // Save to localStorage
        this.saveToLocalStorage();
    }
    
    checkPWAInstall() {
        // Check if app is running as PWA
        if (window.matchMedia('(display-mode: standalone)').matches) {
            console.log('Running as PWA');
        } else if ('BeforeInstallPromptEvent' in window) {
            // Show install button or prompt
            window.addEventListener('beforeinstallprompt', (e) => {
                e.preventDefault();
                this.showInstallPrompt(e);
            });
        }
    }
    
    showInstallPrompt(deferredPrompt) {
        // You could add an install button to your UI here
        console.log('App can be installed');
    }
    
    saveToLocalStorage() {
        localStorage.setItem('tradingAppPortfolio', JSON.stringify(this.portfolio));
        localStorage.setItem('tradingAppWatchlist', JSON.stringify(this.watchlist));
    }
    
    loadFromLocalStorage() {
        const savedPortfolio = localStorage.getItem('tradingAppPortfolio');
        const savedWatchlist = localStorage.getItem('tradingAppWatchlist');
        
        if (savedPortfolio) {
            this.portfolio = JSON.parse(savedPortfolio);
        }
        
        if (savedWatchlist) {
            this.watchlist = JSON.parse(savedWatchlist);
        }
    }
    
    showNotification(message, type = 'success') {
        // Create notification element
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 60px;
            left: 50%;
            transform: translateX(-50%);
            background: ${type === 'success' ? '#10b981' : '#ef4444'};
            color: white;
            padding: 12px 24px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            z-index: 2000;
            animation: slideDown 0.3s ease-out;
        `;
        
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideUp 0.3s ease-in';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 3000);
        
        // Add CSS for animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideDown {
                from { top: -60px; opacity: 0; }
                to { top: 60px; opacity: 1; }
            }
            @keyframes slideUp {
                from { top: 60px; opacity: 1; }
                to { top: -60px; opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    }
}

// Global functions for UI interaction
function showTab(tabName) {
    // Hide all content sections
    document.querySelectorAll('.card').forEach(card => card.style.display = 'none');
    
    // Show selected tab content
    if (tabName === 'dashboard') {
        document.getElementById('marketOverview').style.display = 'block';
        document.getElementById('watchlistCard').style.display = 'block';
        document.getElementById('portfolioCard').style.display = 'block';
        document.getElementById('newsCard').style.display = 'block';
    } else if (tabName === 'markets') {
        document.getElementById('watchlistCard').style.display = 'block';
        document.getElementById('newsCard').style.display = 'block';
    } else if (tabName === 'portfolio') {
        document.getElementById('portfolioCard').style.display = 'block';
    }
    
    // Update active nav item
    document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));
    const navIndex = ['dashboard', 'markets', 'trade', 'portfolio', 'settings'].indexOf(tabName);
    document.querySelectorAll('.nav-item')[navIndex].classList.add('active');
}

function openTradePanel(action) {
    const tradeType = action === 'BUY' ? 'BUY' : 'SELL';
    document.getElementById('tradeType').textContent = `${tradeType} ${app.currentSymbol}`;
    
    // Update button color based on action
    const executeBtn = document.getElementById('executeTradeBtn');
    executeBtn.style.background = action === 'BUY' ? '#10b981' : '#ef4444';
    executeBtn.innerHTML = `<i class="fas fa-check"></i> CONFIRM ${tradeType}`;
    
    // Reset quantity
    document.getElementById('tradeQuantity').value = '10';
    app.updateTradeTotal();
    
    // Show panel
    document.getElementById('tradeOverlay').classList.add('active');
    document.getElementById('tradePanel').classList.add('active');
}

function closeTradePanel() {
    document.getElementById('tradeOverlay').classList.remove('active');
    document.getElementById('tradePanel').classList.remove('active');
}

function setOrderType(type) {
    document.querySelectorAll('.order-type-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    
    // Show/hide limit price input
    document.getElementById('limitPriceSection').style.display = type === 'limit' ? 'block' : 'none';
}

// Initialize app when page loads
let app;
window.addEventListener('DOMContentLoaded', () => {
    app = new TradingApp();
});

// Prevent pull-to-refresh on iOS
document.addEventListener('touchmove', function(e) {
    if (e.scale !== 1) {
        e.preventDefault();
    }
}, { passive: false });