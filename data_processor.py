# tse_lib/data_processor.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class TSEDataProcessor:
    """پردازش و تحلیل داده‌های بورس تهران"""

    def __init__(self):
        pass

    def process_historical_data(self, data, symbol_name):
        """پردازش داده‌های تاریخی"""
        if not data:
            print("❌ داده‌های خالی")
            return None

        print(f"📊 پردازش داده‌های تاریخی {symbol_name}")

        try:
            # تبدیل به DataFrame
            df = pd.DataFrame(data)

            # تشخیص تعداد ستون‌ها و نام‌گذاری مناسب
            num_cols = len(df.columns)
            print(f"🔍 تعداد ستون‌های دریافتی: {num_cols}")

            if num_cols == 6:
                # فرمت ساده: تاریخ، باز، بالا، پایین، بسته، حجم
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            elif num_cols == 7:
                # فرمت با ارزش: تاریخ، باز، بالا، پایین، بسته، حجم، ارزش
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'value']
            elif num_cols == 9:
                # فرمت کامل TSE
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'value', 'count', 'change']
            else:
                print(f"❌ تعداد ستون‌های غیرقابل پردازش: {num_cols}")
                # سعی در استفاده از ستون‌های اصلی
                if num_cols >= 6:
                    df.columns = ['date', 'open', 'high', 'low', 'close', 'volume'] + [f'col_{i}' for i in
                                                                                       range(6, num_cols)]
                    df = df[['date', 'open', 'high', 'low', 'close', 'volume']]  # فقط ستون‌های اصلی
                else:
                    return None

            print(f"✅ ستون‌ها نام‌گذاری شدند: {list(df.columns)}")

            # تبدیل انواع داده‌ها
            df['date'] = pd.to_datetime(df['date'])

            # ستون‌های عددی
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            if 'value' in df.columns:
                numeric_cols.append('value')
            if 'count' in df.columns:
                numeric_cols.append('count')
            if 'change' in df.columns:
                numeric_cols.append('change')

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # تنظیم ایندکس
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            # حذف رکوردهای خالی
            df = df.dropna(subset=['close', 'volume'])

            if df.empty:
                print("❌ پس از تمیز کردن، داده‌ای باقی نمانده")
                return None

            # محاسبه شاخص‌های تکنیکال
            df = self._calculate_technical_indicators(df)

            print(f"✅ {len(df)} رکورد پردازش شد")
            print(f"📅 بازه زمانی: {df.index.min().strftime('%Y-%m-%d')} تا {df.index.max().strftime('%Y-%m-%d')}")

            return df

        except Exception as e:
            print(f"❌ خطا در پردازش داده‌ها: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_technical_indicators(self, df):
        """محاسبه شاخص‌های تکنیکال"""
        try:
            # میانگین متحرک
            df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
            df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()

            # حجم به میلیون
            df['volume_million'] = df['volume'] / 1_000_000

            # RSI (Relative Strength Index)
            df['rsi'] = self._calculate_rsi(df['close'])

            # MACD
            df = self._calculate_macd(df)

            # Bollinger Bands
            df = self._calculate_bollinger_bands(df)

            # درصد تغییر روزانه
            df['daily_return'] = df['close'].pct_change() * 100

            # میانگین حجم
            df['volume_avg_10'] = df['volume'].rolling(window=10, min_periods=1).mean()

            # نسبت حجم به میانگین
            df['volume_ratio'] = df['volume'] / df['volume_avg_10']
            df['volume_ratio'] = df['volume_ratio'].fillna(1)

            # High-Low Range
            df['hl_range'] = ((df['high'] - df['low']) / df['close'] * 100).fillna(0)

            # تمیز کردن NaN ها
            df = df.fillna(method='ffill').fillna(method='bfill')

            print("📊 شاخص‌های تکنیکال محاسبه شد")
            return df

        except Exception as e:
            print(f"⚠️ خطا در محاسبه شاخص‌ها: {str(e)}")
            return df

    def _calculate_rsi(self, prices, period=14):
        """محاسبه RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

            # جلوگیری از تقسیم بر صفر
            loss = loss.replace(0, 0.001)
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            return rsi.fillna(50)

        except Exception as e:
            print(f"⚠️ خطا در محاسبه RSI: {str(e)}")
            return pd.Series([50] * len(prices), index=prices.index)

    def _calculate_macd(self, df, fast=12, slow=26, signal=9):
        """محاسبه MACD"""
        try:
            exp1 = df['close'].ewm(span=fast, min_periods=1).mean()
            exp2 = df['close'].ewm(span=slow, min_periods=1).mean()

            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=signal, min_periods=1).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            return df

        except Exception as e:
            print(f"⚠️ خطا در محاسبه MACD: {str(e)}")
            df['macd'] = 0
            df['macd_signal'] = 0
            df['macd_histogram'] = 0
            return df

    def _calculate_bollinger_bands(self, df, period=20, std_dev=2):
        """محاسبه نوارهای بولینگر"""
        try:
            sma = df['close'].rolling(window=period, min_periods=1).mean()
            std = df['close'].rolling(window=period, min_periods=1).std()

            df['bb_upper'] = sma + (std * std_dev)
            df['bb_middle'] = sma
            df['bb_lower'] = sma - (std * std_dev)

            # موقعیت در کانال بولینگر
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_position'] = df['bb_position'].fillna(0.5).clip(0, 1)

            return df

        except Exception as e:
            print(f"⚠️ خطا در محاسبه Bollinger Bands: {str(e)}")
            df['bb_upper'] = df['close']
            df['bb_middle'] = df['close']
            df['bb_lower'] = df['close']
            df['bb_position'] = 0.5
            return df
