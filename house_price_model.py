
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')


if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

class HousePriceNN(nn.Module):
    """Neural network for house price prediction"""
    def __init__(self, input_dim):
        super(HousePriceNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class HousePricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.city_map = None
        self.device = device
        
    def prepare_features(self, df, is_training=False):
        """Prepare features for training/prediction"""
        df = df.copy()
        features = pd.DataFrame()
        current_year = 2024
        
        if 'city' in df.columns:
            if is_training:
                if 'price' in df.columns:
                    city_avg = df.groupby('city')['price'].mean()
                    mean_price = city_avg.mean()
                    self.city_map = (city_avg / mean_price).to_dict()
                else:
                    self.city_map = {}
                features['city_factor'] = df['city'].map(self.city_map).fillna(1.0)
            else:
                features['city_factor'] = df['city'].map(self.city_map).fillna(1.0)
        
        if 'sqft_living' in df.columns:
            features['sqft_living'] = df['sqft_living'] / 1000
        if 'bedrooms' in df.columns:
            features['bedrooms'] = df['bedrooms']
        if 'bathrooms' in df.columns:
            features['bathrooms'] = df['bathrooms']
        if 'bedrooms' in df.columns and 'bathrooms' in df.columns:
            features['total_rooms'] = df['bedrooms'] + df['bathrooms']
        
        if 'condition' in df.columns:
            features['condition'] = df['condition']
        if 'view' in df.columns:
            features['view'] = df['view']
        if 'waterfront' in df.columns:
            features['waterfront'] = df['waterfront'].astype(float)
        if 'yr_built' in df.columns:
            features['yr_built'] = df['yr_built']
            house_age = current_year - df['yr_built']
            features['house_age'] = house_age.clip(0, 100)
        if 'yr_renovated' in df.columns:
            yr_renovated = df['yr_renovated'].fillna(0)
            is_renovated = (yr_renovated > 0).astype(float)
            features['is_renovated'] = is_renovated
            
            years_since_reno = np.where(
                yr_renovated > 0,
                (current_year - yr_renovated).clip(0, 50),
                50
            )
            features['years_since_reno'] = years_since_reno / 50
            features['renovation_benefit'] = is_renovated * (1 - features['years_since_reno'])
        
        if 'sqft_lot' in df.columns:
            features['sqft_lot'] = df['sqft_lot'] / 1000
        if 'floors' in df.columns:
            features['floors'] = df['floors']
        if 'sqft_above' in df.columns:
            features['sqft_above'] = df['sqft_above'] / 1000
        if 'sqft_basement' in df.columns:
            features['sqft_basement'] = df['sqft_basement'] / 1000
        
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        return features
    
    def train(self, df, target_col='price', epochs=100, batch_size=256, lr=0.001):
        """Train the model"""
        
        print("🔄 Preparing features...")
        X = self.prepare_features(df, is_training=True)
        y = df[target_col].values.astype(np.float32)

        y_log = np.log1p(y)
        self.feature_names = X.columns.tolist()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_log, test_size=0.2, random_state=42
        )
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).reshape(-1, 1).to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        self.model = HousePriceNN(X_train.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_losses.append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        return train_losses, val_losses
    
    def predict(self, input_data):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.eval()
        

        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        X = self.prepare_features(input_data, is_training=False)

        for feature in self.feature_names:
            if feature not in X.columns:
                X[feature] = 0

        X = X[self.feature_names]

        X_scaled = self.scaler.transform(X)

        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        with torch.no_grad():
            pred_log = self.model(X_tensor)
            pred = torch.expm1(pred_log).cpu().numpy()
        
        price = float(pred.flatten()[0])
        
        return {
            'predicted_price': price,
            'lower_bound': price * 0.9,
            'upper_bound': price * 1.1
        }
    
    def save_model(self, path='house_price_model'):
        """Save model - use separate files for different components"""
        if self.model is not None:
            torch.save(self.model.state_dict(), f'{path}_weights.pth')
            
            joblib.dump({
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'city_map': self.city_map
            }, f'{path}_components.pkl')
            
            print(f"✅ Model saved")
    
    def load_model(self, path='house_price_model'):
        """Load model - compatible with PyTorch 2.6+"""
        try:
            self.model = HousePriceNN(len(self.feature_names)).to(self.device)
            self.model.load_state_dict(torch.load(f'{path}_weights.pth', 
                                                  map_location=self.device))
            
            components = joblib.load(f'{path}_components.pkl')
            self.scaler = components['scaler']
            self.feature_names = components['feature_names']
            self.city_map = components.get('city_map', {})
            
            self.model.eval()
            print(f"✅ Model loaded successfully")
            
        except FileNotFoundError:
            print("No saved model found")
            self.model = None