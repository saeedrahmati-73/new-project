import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings

warnings.filterwarnings('ignore')


class PricePredictor:
    def __init__(self):
        self.model = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 10
        self.feature_columns = ['close', 'volume', 'sma_10', 'sma_20', 'rsi', 'macd', 'bb_upper', 'bb_lower',
                                'volume_ratio']

    def create_sequences(self, data, target_column='close'):
        """ایجاد توالی‌های زمانی برای مدل LSTM"""
        try:
            # انتخاب ویژگی‌های موجود
            available_features = [col for col in self.feature_columns if col in data.columns]

            if not available_features:
                print("⚠️ هیچ ویژگی مناسبی پیدا نشد!")
                return None, None

            # آماده‌سازی داده‌ها
            feature_data = data[available_features].values
            target_data = data[target_column].values.reshape(-1, 1)

            # نرمال‌سازی
            feature_scaled = self.scaler_X.fit_transform(feature_data)
            target_scaled = self.scaler_y.fit_transform(target_data)

            # ایجاد توالی‌ها
            X, y = [], []
            for i in range(self.sequence_length, len(feature_scaled)):
                X.append(feature_scaled[i - self.sequence_length:i])
                y.append(target_scaled[i, 0])

            return np.array(X), np.array(y)

        except Exception as e:
            print(f"❌ خطا در ایجاد توالی‌ها: {e}")
            return None, None

    def build_model(self, input_shape):
        """ساخت مدل LSTM"""
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])

            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )

            return model

        except Exception as e:
            print(f"❌ خطا در ساخت مدل: {e}")
            return None

    def calculate_metrics(self, y_true, y_pred):
        """محاسبه معیارهای دقت"""
        try:
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mse)

            # محاسبه MAPE با جلوگیری از تقسیم بر صفر
            non_zero_mask = y_true != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
            else:
                mape = 0

            # محاسبه R²
            r2 = r2_score(y_true, y_pred)

            return {
                'MSE': mse,
                'MAE': mae,
                'RMSE': rmse,
                'MAPE': mape,
                'R²': r2
            }

        except Exception as e:
            print(f"❌ خطا در محاسبه معیارها: {e}")
            return {}

    def predict_prices(self, data, symbol):
        """پیش‌بینی قیمت سهام - متد اصلی"""
        try:
            print(f"🚀 شروع پیش‌بینی برای {symbol}...")

            # بررسی داده‌ها
            if data is None or len(data) < self.sequence_length + 10:
                print("❌ داده‌های کافی برای پیش‌بینی وجود ندارد!")
                return None

            # ایجاد توالی‌ها
            X, y = self.create_sequences(data)
            if X is None or y is None:
                return None

            # تقسیم داده‌ها
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            print(f"📊 آموزش: {len(X_train)} نمونه، تست: {len(X_test)} نمونه")

            # ساخت و آموزش مدل
            self.model = self.build_model((X.shape[1], X.shape[2]))
            if self.model is None:
                return None

            print("🔄 آموزش مدل...")
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )

            history = self.model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )

            # پیش‌بینی
            y_pred_scaled = self.model.predict(X_test, verbose=0)

            # تبدیل به مقیاس اصلی
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            y_test_original = self.scaler_y.inverse_transform(y_test.reshape(-1, 1))

            # محاسبه معیارها
            metrics = self.calculate_metrics(y_test_original.flatten(), y_pred.flatten())

            # پیش‌بینی آینده
            future_predictions = self.predict_future_prices(data, days=5)

            # قیمت فعلی
            current_price = data['close'].iloc[-1]
            next_day_pred = future_predictions[0] if future_predictions else current_price
            predicted_change = ((next_day_pred - current_price) / current_price) * 100

            # نتایج
            results = {
                'symbol': symbol,
                'current_price': current_price,
                'next_day_prediction': next_day_pred,
                'predicted_change': predicted_change,
                'future_predictions': future_predictions,
                'actual_prices': y_test_original.flatten().tolist(),
                'predicted_prices': y_pred.flatten().tolist(),
                'metrics': metrics,
                'training_history': {
                    'loss': history.history['loss'],
                    'val_loss': history.history['val_loss'],
                    'mae': history.history['mae'],
                    'val_mae': history.history['val_mae']
                }
            }

            print(f"✅ پیش‌بینی {symbol} کامل شد!")
            print(f"📈 قیمت فعلی: {current_price:,.0f}")
            print(f"🎯 پیش‌بینی فردا: {next_day_pred:,.0f}")
            print(f"📊 تغییر پیش‌بینی: {predicted_change:+.2f}%")

            return results

        except Exception as e:
            print(f"❌ خطا در پیش‌بینی: {e}")
            return None

    def predict_future_prices(self, data, days=5):
        """پیش‌بینی قیمت‌های آینده"""
        try:
            if self.model is None:
                return []

            # آخرین توالی
            available_features = [col for col in self.feature_columns if col in data.columns]
            feature_data = data[available_features].values
            feature_scaled = self.scaler_X.transform(feature_data)

            last_sequence = feature_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)

            future_prices = []
            current_sequence = last_sequence.copy()

            for _ in range(days):
                # پیش‌بینی
                pred_scaled = self.model.predict(current_sequence, verbose=0)
                pred_price = self.scaler_y.inverse_transform(pred_scaled)[0, 0]
                future_prices.append(pred_price)

                # به‌روزرسانی توالی (ساده‌سازی شده)
                new_features = current_sequence[0, -1, :].copy()
                new_features[0] = pred_scaled[0, 0]  # close price

                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = new_features

            return future_prices

        except Exception as e:
            print(f"❌ خطا در پیش‌بینی آینده: {e}")
            return []


def test_price_predictor():
    """تست کلاس پیش‌بینی"""
    try:
        # import با ساختار صحیح
        from tse_data_collector import TSEDataCollector
        from data_processor import TSEDataProcessor

        print("🧪 تست پیش‌بینی قیمت...")

        collector = TSEDataCollector()
        processor = TSEDataProcessor()
        predictor = PricePredictor()

        # دریافت داده‌ها
        print("📊 دریافت داده‌های فولاد...")
        raw_data = collector.get_historical_data('فولاد', 60)
        if not raw_data:
            print("❌ خطا در دریافت داده‌ها")
            return

        # پردازش داده‌ها
        print("🔄 پردازش داده‌ها...")
        processed_data = processor.process_historical_data(raw_data, 'فولاد')
        if processed_data is None:
            print("❌ خطا در پردازش داده‌ها")
            return

        # پیش‌بینی
        results = predictor.predict_prices(processed_data, 'فولاد')

        if results:
            print("✅ تست موفقیت‌آمیز!")
            print(f"📈 قیمت فعلی: {results['current_price']:,.0f}")
            print(f"🎯 پیش‌بینی فردا: {results['next_day_prediction']:,.0f}")
            print(f"📊 تغییر پیش‌بینی: {results['predicted_change']:+.2f}%")
            print(f"🔢 معیارهای دقت:")
            for key, value in results['metrics'].items():
                print(f"   {key}: {value:.4f}")
        else:
            print("❌ خطا در پیش‌بینی")

    except ImportError as e:
        print(f"❌ خطا در import: {e}")
        print("💡 لطفاً بررسی کنید که فایل‌های زیر موجود باشند:")
        print("   - data_collector.py")
        print("   - data_processor.py")
        print("   و کلاس‌های TSEDataCollector و TSEDataProcessor در آنها تعریف شده باشند")
    except Exception as e:
        print(f"❌ خطا در تست: {e}")


if __name__ == "__main__":
    test_price_predictor()
