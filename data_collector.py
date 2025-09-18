import requests
import urllib3
import os
from datetime import datetime
import time
import pandas as pd

urllib3.disable_warnings()
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)


class TSEDataCollector:
    """کلاس بهبود یافته برای دریافت داده‌های بورس تهران"""

    def __init__(self):
        self.base_url = "https://old.tsetmc.com"
        self.session = requests.Session()
        self.session.proxies = {'http': None, 'https': None}
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'fa,en-US;q=0.7,en;q=0.3',
            'Connection': 'keep-alive',
        })
        self.symbol_cache = {}

    def _make_request(self, url, params=None, timeout=15, retry=2):
        """درخواست امن با retry"""
        for attempt in range(retry):
            try:
                if attempt > 0:
                    time.sleep(1)  # صبر بین تلاش‌ها

                response = self.session.get(url, params=params, timeout=timeout, verify=False)

                if response.status_code == 200:
                    content = response.text.strip()
                    if content and not content.startswith('<'):
                        return content
                    else:
                        if attempt == 0:  # فقط اولین بار چاپ کن
                            print(f"⚠️ پاسخ HTML یا خالی از {url}")
                        return None
                else:
                    print(f"❌ خطای HTTP {response.status_code}: {url}")
                    return None

            except Exception as e:
                if attempt == retry - 1:  # آخرین تلاش
                    print(f"💥 خطا در درخواست {url}: {e}")
                return None

    def search_symbol(self, symbol_name):
        """جستجوی نماد با انتخاب هوشمند"""
        print(f"🔍 جستجوی نماد: {symbol_name}")

        url = f"{self.base_url}/tsev2/data/search.aspx"
        params = {'skey': symbol_name}

        data = self._make_request(url, params)
        if not data:
            return None

        symbols = []
        lines = data.split(';')

        for line in lines:
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 10:
                    symbol_info = {
                        'symbol': parts[0],
                        'name': parts[1],
                        'id': parts[2],
                        'id2': parts[3],
                        'id3': parts[4],
                        'id4': parts[5],
                        'status1': parts[6],
                        'status2': parts[7],
                        'status3': parts[8],
                        'status4': parts[9],
                        'market': parts[10] if len(parts) > 10 else 'N/A'
                    }
                    symbols.append(symbol_info)
                    self.symbol_cache[parts[0]] = symbol_info

        print(f"✅ {len(symbols)} نماد یافت شد")
        return symbols

    def get_best_symbol_match(self, symbol_name):
        """انتخاب بهترین نماد مطابق"""
        symbols = self.search_symbol(symbol_name)
        if not symbols:
            return None

        # اولویت‌بندی انتخاب نماد:
        # 1. نماد دقیقاً همان نام
        # 2. نماد شروع شده با آن نام
        # 3. بازار اصلی (N1) در اولویت
        # 4. اولین نماد

        exact_matches = [s for s in symbols if s['symbol'].lower() == symbol_name.lower()]
        if exact_matches:
            # اگر چند نماد دقیق بود، بازار N1 را ترجیح بده
            main_market = [s for s in exact_matches if s['market'] == 'N1']
            if main_market:
                return main_market[0]
            return exact_matches[0]

        # نماد شروع شده با آن نام
        prefix_matches = [s for s in symbols if s['symbol'].lower().startswith(symbol_name.lower())]
        if prefix_matches:
            main_market = [s for s in prefix_matches if s['market'] == 'N1']
            if main_market:
                return main_market[0]
            return prefix_matches[0]

        # اولین نماد
        return symbols[0]

    def get_historical_data(self, symbol_name, days=30):
        """دریافت داده‌های تاریخی"""
        print(f"📊 دریافت تاریخچه {symbol_name} - {days} روز اخیر")

        symbol_info = self.get_best_symbol_match(symbol_name)
        if not symbol_info:
            print("❌ نماد یافت نشد")
            return None

        symbol_id = symbol_info['id']
        url = f"{self.base_url}/tsev2/data/InstTradeHistory.aspx"
        params = {'i': symbol_id, 'Top': str(days), 'A': '0'}

        data = self._make_request(url, params)
        if not data:
            return None

        historical_data = []
        entries = data.split(';')

        for entry in entries:
            if entry.strip():
                parts = entry.split('@')
                if len(parts) >= 8:
                    try:
                        record = {
                            'date': parts[0],
                            'high': float(parts[1]),
                            'low': float(parts[2]),
                            'open': float(parts[3]),
                            'close': float(parts[4]),
                            'yesterday_close': float(parts[5]),
                            'value': float(parts[6]),
                            'volume': int(float(parts[7])),
                            'count': int(parts[8]) if len(parts) > 8 else 0
                        }
                        historical_data.append(record)
                    except (ValueError, IndexError):
                        continue

        print(f"✅ {len(historical_data)} رکورد تاریخی دریافت شد")
        return historical_data

    def get_client_type_data(self, symbol_name, days=10):
        """دریافت اطلاعات نوع سرمایه‌گذار"""
        print(f"👥 دریافت نوع سرمایه‌گذاران {symbol_name}")

        symbol_info = self.get_best_symbol_match(symbol_name)
        if not symbol_info:
            print("❌ نماد یافت نشد")
            return None

        symbol_id = symbol_info['id']
        url = f"{self.base_url}/tsev2/data/clienttype.aspx"
        params = {'i': symbol_id}

        data = self._make_request(url, params)
        if not data:
            return None

        client_data = []
        lines = data.split(';')

        for line in lines[:days]:  # فقط چند روز اخیر
            if line.strip():
                parts = line.split(',')
                if len(parts) >= 12:
                    try:
                        record = {
                            'date': parts[0],
                            'individual_buy_count': int(parts[1]),
                            'corporate_buy_count': int(parts[2]),
                            'individual_sell_count': int(parts[3]),
                            'corporate_sell_count': int(parts[4]),
                            'individual_buy_volume': int(parts[5]),
                            'corporate_buy_volume': int(parts[6]),
                            'individual_sell_volume': int(parts[7]),
                            'corporate_sell_volume': int(parts[8]),
                            'individual_buy_value': float(parts[9]),
                            'corporate_buy_value': float(parts[10]),
                            'individual_sell_value': float(parts[11]),
                            'corporate_sell_value': float(parts[12]) if len(parts) > 12 else 0
                        }
                        client_data.append(record)
                    except (ValueError, IndexError):
                        continue

        print(f"✅ {len(client_data)} رکورد نوع سرمایه‌گذار دریافت شد")
        return client_data

    def get_symbol_summary(self, symbol_name):
        """خلاصه اطلاعات یک نماد"""
        print(f"\n🎯 خلاصه اطلاعات {symbol_name}")

        # اطلاعات پایه
        symbol_info = self.get_best_symbol_match(symbol_name)
        if not symbol_info:
            print("❌ نماد یافت نشد")
            return None

        print(f"📊 {symbol_info['symbol']}: {symbol_info['name']}")
        print(f"   🆔 ID: {symbol_info['id']}")
        print(f"   🏪 بازار: {symbol_info['market']}")

        # آخرین قیمت
        hist_data = self.get_historical_data(symbol_name, 5)
        if hist_data and len(hist_data) > 0:
            latest = hist_data[0]
            change = latest['close'] - latest['yesterday_close']
            change_percent = (change / latest['yesterday_close']) * 100

            print(f"   💰 قیمت پایانی: {latest['close']:,.0f}")
            print(f"   📈 تغییر: {change:+,.0f} ({change_percent:+.1f}%)")
            print(f"   📊 حجم: {latest['volume']:,}")
            print(f"   💵 ارزش: {latest['value']:,.0f}")
            print(f"   📅 تاریخ: {latest['date']}")

        # خالص نوع سرمایه‌گذار
        client_data = self.get_client_type_data(symbol_name, 1)
        if client_data and len(client_data) > 0:
            latest_client = client_data[0]
            ind_net = latest_client['individual_buy_volume'] - latest_client['individual_sell_volume']
            corp_net = latest_client['corporate_buy_volume'] - latest_client['corporate_sell_volume']

            print(f"   👤 خالص حقیقی: {ind_net:,}")
            print(f"   🏢 خالص حقوقی: {corp_net:,}")

        return symbol_info

def collect_stock_data(symbol, days=90):
    """
    Collects historical stock data for a given symbol.
    This function is a wrapper around the TSEDataCollector class to be used by the GUI.
    """
    collector = TSEDataCollector()
    data = collector.get_historical_data(symbol, days=days)
    if data:
        df = pd.DataFrame(data)
        # The GUI expects specific column names. Let's rename them.
        df = df.rename(columns={
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
        # Convert Date to datetime objects
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
        # Set Date as index
        df = df.set_index('Date')
        # The main app expects columns in a certain order, and specific columns.
        # Let's ensure we have the columns it expects.
        # The GUI uses 'Close', 'Low', 'High'. The predictor uses more.
        # Let's provide what the GUI expects, and the predictor can use what it needs.
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return None

# تست بهبود یافته
def test_improved_library():
    print("🧪 تست کتابخانه بهبود یافته TSE\n")

    collector = TSEDataCollector()

    # تست نمادهای مختلف
    symbols = ['فولاد', 'شپنا', 'پترو', 'خودرو', 'کگل']

    for symbol in symbols:
        print("=" * 60)
        collector.get_symbol_summary(symbol)
        time.sleep(1)  # صبر بین درخواست‌ها


if __name__ == "__main__":
    test_improved_library()
