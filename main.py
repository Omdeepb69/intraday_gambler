import yfinance as yf
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NextDayTradingPredictor:
    def __init__(self, ticker, seq_length=30, hidden_dim=128, num_layers=3):
        self.ticker = ticker
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.scaler = StandardScaler()
        self.model = None
        self.df = None
        self.ist = pytz.timezone('Asia/Kolkata')
        
    def is_market_time(self):
        now = datetime.now(self.ist)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        is_weekday = now.weekday() < 5
        is_trading_hours = market_open <= now <= market_close
        
        return is_weekday and is_trading_hours
    
    def get_next_trading_day(self):
        now = datetime.now(self.ist)
        next_day = now + timedelta(days=1)
        
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
            
        return next_day.date()
    
    def is_weekend(self):
        """Check if today is Saturday or Sunday"""
        now = datetime.now(self.ist)
        return now.weekday() >= 5  
    
    def should_make_prediction(self):
        """Check if we should make a prediction today"""
        now = datetime.now(self.ist)
        current_day = now.weekday()  
        
        if current_day >= 5:
            return False, "Weekend - Indian stock market is closed on weekends"
        
        return True, "Trading day - prediction available"

    def fetch_data(self):
        try:
            if not self.ticker.endswith('.NS') and not self.ticker.endswith('.BO'):
                ticker_symbol = f"{self.ticker}.NS"
            else:
                ticker_symbol = self.ticker
                
            print(f"Fetching historical data for {ticker_symbol}...")
            
            self.df = yf.download(ticker_symbol, period="2y", interval="1d")
            
            if self.df.empty:
                self.df = yf.download(ticker_symbol, period="1y", interval="1d")
                
            if self.df.empty:
                raise ValueError(f"No data found for ticker {ticker_symbol}")
                
            self.df['Returns'] = self.df['Close'].pct_change()
            self.df['Volatility'] = self.df['Returns'].rolling(window=20).std()
            self.df['RSI'] = self.calculate_rsi(self.df['Close'])
            self.df['MA_20'] = self.df['Close'].rolling(window=20).mean()
            self.df['MA_50'] = self.df['Close'].rolling(window=50).mean()
            self.df['Volume_Ratio'] = self.df['Volume'] / self.df['Volume'].rolling(window=20).mean()
            self.df['Price_Range'] = (self.df['High'] - self.df['Low']) / self.df['Close']
            
            self.df = self.df.fillna(method='ffill').fillna(method='bfill')
            
            print(f"Data loaded: {len(self.df)} trading days")
            print(f"Date range: {self.df.index[0].date()} to {self.df.index[-1].date()}")
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
        return True
    
    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def prepare_data(self):
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility', 'RSI', 'MA_20', 'MA_50', 'Volume_Ratio', 'Price_Range']
        
        feature_data = self.df[features].values
        scaled_data = self.scaler.fit_transform(feature_data)
        
        X, y_high, y_low, y_close = [], [], [], []
        
        for i in range(self.seq_length, len(scaled_data)):
            X.append(scaled_data[i-self.seq_length:i])
            y_high.append(scaled_data[i, 1])  
            y_low.append(scaled_data[i, 2])   
            y_close.append(scaled_data[i, 3]) 
        
        X = np.array(X)
        y_high = np.array(y_high)
        y_low = np.array(y_low)
        y_close = np.array(y_close)
        
        split_idx = int(len(X) * 0.85)
        
        self.X_train = torch.FloatTensor(X[:split_idx]).to(device)
        self.y_train_high = torch.FloatTensor(y_high[:split_idx]).to(device)
        self.y_train_low = torch.FloatTensor(y_low[:split_idx]).to(device)
        self.y_train_close = torch.FloatTensor(y_close[:split_idx]).to(device)
        
        self.X_test = torch.FloatTensor(X[split_idx:]).to(device)
        self.y_test_high = torch.FloatTensor(y_high[split_idx:]).to(device)
        self.y_test_low = torch.FloatTensor(y_low[split_idx:]).to(device)
        self.y_test_close = torch.FloatTensor(y_close[split_idx:]).to(device)
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        
    def build_model(self):
        class MultiTargetLSTM(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers):
                super(MultiTargetLSTM, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                                  batch_first=True, dropout=0.3)
                self.dropout = nn.Dropout(0.3)
                
                self.fc_high = nn.Linear(hidden_dim, 1)
                self.fc_low = nn.Linear(hidden_dim, 1)
                self.fc_close = nn.Linear(hidden_dim, 1)
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.dropout(out[:, -1, :])
                
                high_pred = self.fc_high(out)
                low_pred = self.fc_low(out)
                close_pred = self.fc_close(out)
                
                return high_pred, low_pred, close_pred
        
        input_dim = self.X_train.shape[2]
        self.model = MultiTargetLSTM(input_dim, self.hidden_dim, self.num_layers).to(device)
        
    def train_model(self, epochs=200, lr=0.001):
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        
        print("Training model for next-day predictions...")
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            high_pred, low_pred, close_pred = self.model(self.X_train)
            
            loss_high = criterion(high_pred.squeeze(), self.y_train_high)
            loss_low = criterion(low_pred.squeeze(), self.y_train_low)
            loss_close = criterion(close_pred.squeeze(), self.y_train_close)
            
            total_loss = loss_high + loss_low + loss_close
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            self.model.eval()
            with torch.no_grad():
                test_high, test_low, test_close = self.model(self.X_test)
                
                test_loss_high = criterion(test_high.squeeze(), self.y_test_high)
                test_loss_low = criterion(test_low.squeeze(), self.y_test_low)
                test_loss_close = criterion(test_close.squeeze(), self.y_test_close)
                
                test_total_loss = test_loss_high + test_loss_low + test_loss_close
            
            scheduler.step(test_total_loss)
            
            if test_total_loss < best_loss:
                best_loss = test_total_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
            
            if (epoch + 1) % 25 == 0:
                print(f'Epoch [{epoch+1}/{epochs}]')
                print(f'Train Loss: {total_loss.item():.6f}, Test Loss: {test_total_loss.item():.6f}')
                print(f'High: {loss_high.item():.6f}, Low: {loss_low.item():.6f}, Close: {loss_close.item():.6f}')
        
        self.model.load_state_dict(torch.load('best_model.pth'))
        
    def predict_next_day(self):
        self.model.eval()
        
        with torch.no_grad():
            latest_sequence = self.X_test[-1:] 
            
            high_pred, low_pred, close_pred = self.model(latest_sequence)
            
            predictions = np.array([[0, high_pred.item(), low_pred.item(), close_pred.item()] + [0]*7])
            actual_predictions = self.scaler.inverse_transform(predictions)
            
            predicted_high = actual_predictions[0, 1]
            predicted_low = actual_predictions[0, 2]
            predicted_close = actual_predictions[0, 3]
            
            current_close = float(self.df['Close'].iloc[-1])
            
            return {
                'current_price': current_close,
                'predicted_high': predicted_high,
                'predicted_low': predicted_low,
                'predicted_close': predicted_close,
                'expected_return': ((predicted_close - current_close) / current_close) * 100,
                'trading_range': predicted_high - predicted_low,
                'risk_reward_ratio': (predicted_high - current_close) / (current_close - predicted_low) if current_close > predicted_low else 0
            }
    
    def generate_trading_signals(self, predictions):
        current_price = predictions['current_price']
        pred_high = predictions['predicted_high']
        pred_low = predictions['predicted_low']
        pred_close = predictions['predicted_close']
        expected_return = predictions['expected_return']
        
        signals = {
            'action': 'HOLD',
            'entry_price': current_price,
            'target_price': pred_close,
            'stop_loss': pred_low * 0.98,
            'profit_potential': 0,
            'risk_level': 'MEDIUM'
        }
        
        if expected_return > 2:
            signals['action'] = 'BUY'
            signals['target_price'] = pred_high * 0.95
            signals['profit_potential'] = ((signals['target_price'] - current_price) / current_price) * 100
            signals['risk_level'] = 'LOW' if expected_return > 5 else 'MEDIUM'
            
        elif expected_return < -2:
            signals['action'] = 'SELL/SHORT'
            signals['target_price'] = pred_low * 1.05
            signals['profit_potential'] = ((current_price - signals['target_price']) / current_price) * 100
            signals['risk_level'] = 'HIGH'
            
        return signals
    
    def plot_next_day_prediction(self, predictions):
        recent_data = self.df.tail(30)
        
        plt.figure(figsize=(16, 12))
        
        plt.subplot(2, 2, 1)
        plt.plot(recent_data.index, recent_data['Close'], 'b-', linewidth=2, label='Historical Close')
        plt.plot(recent_data.index, recent_data['High'], 'g--', alpha=0.7, label='Historical High')
        plt.plot(recent_data.index, recent_data['Low'], 'r--', alpha=0.7, label='Historical Low')
        
        next_day = self.get_next_trading_day()
        plt.axvline(x=recent_data.index[-1], color='gray', linestyle=':', alpha=0.7)
        
        future_x = [recent_data.index[-1], pd.Timestamp(next_day)]
        plt.plot(future_x, [predictions['current_price'], predictions['predicted_close']], 
                'orange', linewidth=3, marker='o', label='Predicted Close')
        plt.plot(future_x, [predictions['current_price'], predictions['predicted_high']], 
                'green', linewidth=2, marker='^', label='Predicted High')
        plt.plot(future_x, [predictions['current_price'], predictions['predicted_low']], 
                'red', linewidth=2, marker='v', label='Predicted Low')
        
        plt.title(f'{self.ticker} - Next Trading Day Prediction', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price (₹)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 2)
        labels = ['Current Price', 'Predicted Low', 'Predicted Close', 'Predicted High']
        values = [predictions['current_price'], predictions['predicted_low'], 
                 predictions['predicted_close'], predictions['predicted_high']]
        colors = ['blue', 'red', 'orange', 'green']
        
        bars = plt.bar(labels, values, color=colors, alpha=0.7)
        plt.title('Price Levels Comparison', fontweight='bold')
        plt.ylabel('Price (₹)')
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                    f'₹{value:.2f}', ha='center', fontweight='bold')
        
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 3)
        profit_scenarios = {
            'Conservative (Low)': ((predictions['predicted_low'] - predictions['current_price']) / predictions['current_price']) * 100,
            'Expected (Close)': predictions['expected_return'],
            'Optimistic (High)': ((predictions['predicted_high'] - predictions['current_price']) / predictions['current_price']) * 100
        }
        
        colors_profit = ['red' if v < 0 else 'green' for v in profit_scenarios.values()]
        bars = plt.bar(profit_scenarios.keys(), profit_scenarios.values(), color=colors_profit, alpha=0.7)
        plt.title('Profit/Loss Scenarios (%)', fontweight='bold')
        plt.ylabel('Return (%)')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        for bar, value in zip(bars, profit_scenarios.values()):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(profit_scenarios.values())*0.05 if value > 0 else min(profit_scenarios.values())*0.05), 
                    f'{value:.1f}%', ha='center', fontweight='bold')
        
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        recent_volume = recent_data['Volume'].tail(10)
        plt.plot(recent_volume.index, recent_volume.values, 'purple', linewidth=2, marker='o')
        plt.title('Recent Volume Trend', fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def display_trading_recommendations(self, predictions, signals):
        next_day = self.get_next_trading_day()
        market_status = "OPEN" if self.is_market_time() else "CLOSED"
        now = datetime.now(self.ist)
        current_day = now.strftime('%A') 
        
        print("\n" + "="*80)
        print(f"🔥 NEXT DAY TRADING PREDICTIONS FOR {self.ticker} 🔥")
        print("="*80)
        print(f"📅 Today: {current_day}, {now.strftime('%Y-%m-%d')}")
        print(f"📅 Next Trading Day: {next_day}")
        print(f"🕘 Market Status: {market_status}")
        print(f"⏰ Prediction Time: {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
        
        if now.weekday() == 4:  
            print(f"📝 Note: Predicting for Monday as weekend markets are closed")
        
        print(f"\n📊 PRICE PREDICTIONS:")
        print(f"Current Price: ₹{predictions['current_price']:.2f}")
        print(f"Predicted HIGH: ₹{predictions['predicted_high']:.2f}")
        print(f"Predicted LOW:  ₹{predictions['predicted_low']:.2f}")
        print(f"Predicted CLOSE: ₹{predictions['predicted_close']:.2f}")
        
        print(f"\n💰 PROFIT ANALYSIS:")
        print(f"Expected Return: {predictions['expected_return']:.2f}%")
        print(f"Trading Range: ₹{predictions['trading_range']:.2f}")
        print(f"Risk-Reward Ratio: {predictions['risk_reward_ratio']:.2f}")
        
        print(f"\n🎯 TRADING RECOMMENDATION:")
        print(f"Action: {signals['action']}")
        print(f"Entry Price: ₹{signals['entry_price']:.2f}")
        print(f"Target Price: ₹{signals['target_price']:.2f}")
        print(f"Stop Loss: ₹{signals['stop_loss']:.2f}")
        print(f"Profit Potential: {signals['profit_potential']:.2f}%")
        print(f"Risk Level: {signals['risk_level']}")
        
        if signals['action'] == 'BUY':
            print(f"\n🟢 BULLISH SIGNAL - MONEY MAKING STRATEGY:")
            print(f"✅ Buy at market open or on dips near ₹{predictions['current_price']:.2f}")
            print(f"✅ First target: ₹{predictions['predicted_close']:.2f}")
            print(f"✅ Final target: ₹{predictions['predicted_high']:.2f}")
            print(f"✅ Stop loss: ₹{signals['stop_loss']:.2f}")
            
        elif signals['action'] == 'SELL/SHORT':
            print(f"\n🔴 BEARISH SIGNAL - SHORT SELLING STRATEGY:")
            print(f"⚠️ Sell existing positions or short near ₹{predictions['current_price']:.2f}")
            print(f"⚠️ Target: ₹{predictions['predicted_low']:.2f}")
            print(f"⚠️ Cover shorts above: ₹{signals['stop_loss']:.2f}")
            
        else:
            print(f"\n🟡 NEUTRAL SIGNAL - WAIT AND WATCH:")
            print(f"📈 Monitor price action around ₹{predictions['current_price']:.2f}")
            print(f"📈 Enter on breakout above ₹{predictions['predicted_high']:.2f}")
            print(f"📉 Or short below ₹{predictions['predicted_low']:.2f}")
        
        print("="*80)

def main():
    tickers = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC", "HINDUNILVR", "KOTAKBANK", "BAJFINANCE"]
    
    predictor_temp = NextDayTradingPredictor("RELIANCE")  # Temporary instance for checking
    can_predict, reason = predictor_temp.should_make_prediction()
    
    if not can_predict:
        print("\n" + "="*80)
        print("🚫 PREDICTION NOT AVAILABLE")
        print(datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S IST'))
        print("="*80)
        print(f"Reason: {reason}")
        print("📅 Indian stock market is closed on weekends (Saturday & Sunday)")
        print("💡 Please run this script on weekdays (Monday-Friday) for predictions")
        print("🕘 Trading hours: 9:15 AM to 3:30 PM IST")
        print("="*80)
        return
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Analyzing: {ticker}")
        print('='*60)
        
        predictor = NextDayTradingPredictor(ticker)
        
        if predictor.fetch_data():
            predictor.prepare_data()
            predictor.build_model()
            predictor.train_model(epochs=150)
            
            predictions = predictor.predict_next_day()
            signals = predictor.generate_trading_signals(predictions)
            
            predictor.plot_next_day_prediction(predictions)
            predictor.display_trading_recommendations(predictions, signals)
            break
        else:
            print(f"Failed to fetch data for {ticker}, trying next...")
            continue
    else:
        print("Failed to fetch data for all tickers.")

if __name__ == "__main__":
    main()
