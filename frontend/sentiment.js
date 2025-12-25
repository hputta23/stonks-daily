class FinancialSentiment {
    constructor() {
        this.dictionary = {
            "up": 1, "rise": 1, "rises": 1, "rising": 1, "rose": 1, "surge": 2, "surges": 2, "surging": 2, "jump": 2, "jumps": 2, "gain": 1, "gains": 1, "gaining": 1, "profit": 1, "profits": 1, "profitable": 1, "bull": 1, "bullish": 1, "buy": 1, "strong": 1, "growth": 1, "record": 1, "high": 1, "higher": 1, "optimism": 1, "positive": 1, "upgrade": 1,
            "down": -1, "fall": -1, "falls": -1, "falling": -1, "fell": -1, "drop": -1, "drops": -1, "dropping": -1, "decline": -1, "declines": -1, "declining": -1, "loss": -1, "losses": -1, "lost": -1, "bear": -1, "bearish": -1, "sell": -1, "weak": -1, "weakness": -1, "crash": -2, "plunge": -2, "plunges": -2, "low": -1, "lower": -1, "negative": -1, "downgrade": -1, "warning": -1, "risk": -1
        };
    }

    analyze(text) {
        if (!text) return { score: 0, type: 'neutral' };

        const words = text.toLowerCase().match(/\b\w+\b/g) || [];
        let score = 0;

        words.forEach(word => {
            if (this.dictionary[word]) {
                score += this.dictionary[word];
            }
        });

        return {
            score: score,
            type: score > 0 ? 'positive' : score < 0 ? 'negative' : 'neutral'
        };
    }
}
