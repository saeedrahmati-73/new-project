# new-project
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time        
import logging
import warnings
import multiprocessing as mp
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import requests
from retrying import retry
from persiantools.jdatetime import JalaliDate
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==== ۱) خنثی‌سازی پراکسی‌های محیطی ====
for _v in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_v, None)
os.environ["NO_PROXY"] = "*"

# ==== ۲) تنظیم لاگینگ ====
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def fetch_stock_raw(symbol: str) -> str:
    """
    درخواست HTTP به TSETMC می‌زند و متن خام پاسخ را برمی‌گرداند.
    """
    url = f"http://www.tsetmc.com/Loader.aspx?ParTree=151311&i={symbol}"
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/115.0.0.0 Safari/537.36'
        ),
        'Referer': 'http://www.tsetmc.com/Loader.aspx'
    }

    logger.debug("Sending HTTP GET → %s", url)
    resp = requests.get(url, headers=headers, timeout=10)
    logger.debug(
        "HTTP %s %s → status=%s",
        resp.request.method,
        resp.url,
        resp.status_code
    )
    # لاگ ۵۰۰ کاراکتر اول متن پاسخ برای دیباگ
    logger.debug("RAW RESPONSE TEXT (first 500 chars):%s", resp.text[:500])

    resp.raise_for_status()
    return resp.text


class TehranStockAPI:
    """
    کلاس API برای دریافت داده خام از TSETMC و جستجوی نمادها
    با غیرفعال کردن پراکسی و مکانیزم Retry
    """
    def __init__(self):
        self.base_url = "http://www.tsetmc.com/tsev2/data"
        # ساخت Session بدون اتکا به پراکسی محیطی
        self.session = requests.Session()
        self.session.trust_env = False
        self.session.proxies.clear()

        # اضافه کردن Retry به Session
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # هدرهای نمونه
        self.headers_list = [
            {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                              'AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/121.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'fa-IR,fa;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        ]

        # کش اولیه‌ی نمادها
        self.known_symbols = {
            'فولاد': '46348559193224090',
            'خودرو': '65883838195688438',
            'شپنا': '778253364357513',
            'پارس': '35700344742885862',
            'کگل': '17635693434652166',
            'وبملت': '35425587644337450',
            'فملی': '43685683301925375'
        }

    def search_symbol(self, symbol_name: str) -> str | None:
        """جستجوی کد نماد با fallback به کش محلی"""
        logger.info("🔍 جستجوی نماد: %s", symbol_name)
        if symbol_name in self.known_symbols:
            code = self.known_symbols[symbol_name]
            logger.info("✅ کد از کش: %s", code)
            return code

        try:
            headers = np.random.choice(self.headers_list)
            url = f"{self.base_url}/search.aspx"
            params = {'skey': symbol_name}
            resp = self.session.get(
                url, params=params, headers=headers,
                timeout=10, allow_redirects=True
            )
            if resp.status_code == 200 and resp.text.strip():
                part = resp.text.split(';')[0]
                if ',' in part:
                    code = part.split(',')[0]
                    self.known_symbols[symbol_name] = code
                    logger.info("✅ کد آنلاین یافت شد: %s", code)
                    return code
        except Exception as e:
            logger.error("❌ خطا در search_symbol: %s", e)

        logger.error("❌ نماد یافت نشد")
        return None

    def get_stock_data(self, symbol_code: str) -> pd.DataFrame | None:
        """دریافت داده خام سهام از API TSETMC"""
        if not symbol_code:
            logger.error("❌ نماد معتبر نیست")
            return None

        logger.info("📊 درخواست داده برای کد: %s", symbol_code)
        try:
            headers = np.random.choice(self.headers_list)
            url = f"{self.base_url}/Export-txt.aspx"
            params = {'i': symbol_code, 't': 'i', 'a': '0'}
            # تاخیر تصادفی برای جلوگیری از بلاک شدن
            time.sleep(1 + np.random.random())
            resp = self.session.get(
                url, params=params, headers=headers,
                timeout=15, allow_redirects=True
            )

            logger.debug(
                "HTTP %s %s → status=%s",
                resp.request.method,
                resp.url,
                resp.status_code
            )
            if resp.status_code != 200:
                logger.error("❌ کد وضعیت HTTP: %s", resp.status_code)
                return None

            text = resp.text
            # اگر HTML برگردانده شد یعنی احتمالاً بلاک شده‌ایم
            if '<html>' in text.lower():
                logger.error("❌ دسترسی مسدود شد یا صفحه HTML بازگشت داده شد")
                return None

            return self._parse_stock_data(text)

        except Exception as e:
            logger.error("❌ خطا در get_stock_data: %s", e)
            return None

    def _parse_stock_data(self, raw: str) -> pd.DataFrame | None:
        """تبدیل متن خام API به DataFrame"""
        try:
            if not raw.strip():
                logger.error("❌ داده خام برای پارس کردن خالی است")
                return None

            logger.debug("Parsing raw data (first 200 chars):%s", raw[:200])
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            data_list = []

            for line in lines:
                parts = line.split(',')
                if len(parts) < 6:
                    continue

                date_str = parts[0]
                # اعتبارسنجی ساده تاریخ
                if len(date_str) != 8 or not date_str.isdigit():
                    continue
                y, m, d = map(int, (date_str[:4], date_str[4:6], date_str[6:8]))
                if not (1300 <= y <= 1500 and 1 <= m <= 12 and 1 <= d <= 31):
                    continue

                try:
                    dt = datetime(y, m, d)
                    open_, high, low, close = map(float, parts[1:5])
                    volume = int(float(parts[5])) if parts[5].strip() else 0
                except ValueError:
                    continue

                data_list.append({
                    'Date': dt,
                    'Open': open_,
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': volume
                })

            if not data_list:
                logger.error("❌ هیچ رکورد معتبری پس از پارس نشد")
                return None

            df = pd.DataFrame(data_list)
            df = df.drop_duplicates('Date').sort_values('Date').reset_index(drop=True)
            logger.info("✅ %d رکورد پارس و آماده شد", len(df))
            return df

        except Exception as e:
            logger.error("❌ خطا در _parse_stock_data: %s", e)
            logger.debug("RAW CONTENT (200 chars): %r", raw[:200])
            return None


class TehranStockDataManager:
    """
    مدیریت کش و ترکیب داده خام با آماده‌سازی،
    پشتیبانی آفلاین/آنلاین و داده نمونه
    """
    def __init__(self, cache_dir="cache", cache_expiry_days=7):
        self.cache_dir = cache_dir
        self.cache_expiry_days = cache_expiry_days
        self.api = TehranStockAPI()
        self._ensure_cache_dir()
        logger.info("✅ TehranStockDataManager آماده شد")

    def _ensure_cache_dir(self):
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info("📁 پوشه کش ایجاد شد: %s", self.cache_dir)

    def _cache_path(self, symbol: str, days: int) -> str:
        return os.path.join(self.cache_dir, f"{symbol}_{days}.parquet")

    def _is_cache_valid(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))).days
        return age <= self.cache_expiry_days

    def _load_from_cache(self, symbol: str, days: int) -> pd.DataFrame | None:
        path = self._cache_path(symbol, days)
        if not os.path.exists(path):
            return None
        try:
            if self._is_cache_valid(path):
                df = pd.read_parquet(path, engine='fastparquet')
                logger.info("✅ بارگذاری از کش: %s (%d رکورد)", path, len(df))
                return df
        except Exception as e:
            logger.error("❌ خطا در load_from_cache: %s", e)
        return None

    def _save_to_cache(self, symbol: str, days: int, df: pd.DataFrame):
        path = self._cache_path(symbol, days)
        try:
            # تلاش ذخیره با fastparquet
            df.to_parquet(path, engine='fastparquet', compression='snappy')
            logger.info("💾 ذخیره در کش (fastparquet): %s", path)
        except Exception as e_fast:
            logger.warning("⚠️ ذخیره با fastparquet موفق نبود: %s", e_fast)
            try:
                df.to_parquet(path, engine='pyarrow', compression='snappy')
                logger.info("💾 ذخیره در کش (pyarrow): %s", path)
            except Exception as e_arrow:
                logger.error("❌ ذخیره کش با pyarrow هم ناموفق بود: %s", e_arrow)
                try:
                    csv_path = path.replace('.parquet', '.csv')
                    df.to_csv(csv_path, index=False)
                    logger.info("💾 ذخیره اضطراری در CSV: %s", csv_path)
                except Exception as e_csv:
                    logger.error("❌ ذخیره اضطراری CSV هم ناموفق بود: %s", e_csv)

    def clear_cache(self, symbol: str | None = None):
        try:
            files = os.listdir(self.cache_dir)
            for f in files:
                if symbol is None or f.startswith(f"{symbol}_"):
                    os.remove(os.path.join(self.cache_dir, f))
            logger.info("🗑️ کش %s پاک شد", symbol or "همه")
        except Exception as e:
            logger.error("❌ خطا در clear_cache: %s", e)

    def _load_from_api(self, symbol: str, days: int) -> pd.DataFrame | None:
        code = self.api.search_symbol(symbol)
        if not code:
            return None
        df = self.api.get_stock_data(code)
        if df is None or df.empty:
            return None
        return df.tail(days)

    @retry(stop_max_attempt_number=2,
           wait_exponential_multiplier=300,
           wait_exponential_max=3000)
    def get_stock_data(self, symbol: str, days: int = 300, online: bool = True) -> pd.DataFrame | None:
        logger.info(
            "📊 درخواست %d روز برای %s - حالت: %s",
            days, symbol, "آنلاین" if online else "آفلاین"
        )

        # ۱) تلاش از کش
        cached = self._load_from_cache(symbol, days)
        if cached is not None and len(cached) >= 10:
            return cached

        # ۲) روش آنلاین
        if online:
            df_api = self._load_from_api(symbol, days)
            if df_api is not None and len(df_api) >= 10:
                df_prep = self._prepare_data(df_api)
                self._save_to_cache(symbol, days, df_prep)
                return df_prep
            logger.warning("⚠️ دریافت آنلاین ناموفق، استفاده از داده نمونه")

        else:
            logger.info("⚠️ حالت آفلاین: از کش یا نمونه استفاده می‌شود")

        # ۳) داده نمونه
        sample = self._get_sample_data(symbol, days)
        if sample is not None:
            df_prep = self._prepare_data(sample)
            self._save_to_cache(symbol, days, df_prep)
            return df_prep

        logger.error("❌ هیچ داده‌ای یافت نشد")
        return None

    def _get_sample_data(self, symbol: str, days: int) -> pd.DataFrame | None:
        """تولید داده تصادفی برای شرایط fallback"""
        try:
            end = datetime.now()
            start = end - timedelta(days=int(days * 1.5))
            all_dates = pd.date_range(start, end, freq='D')
            biz = all_dates[all_dates.weekday < 5]
            dates = biz[-days:] if len(biz) >= days else biz

            base = 1000 + (hash(symbol) % 10000)
            n = len(dates)
            np.random.seed(hash(symbol) % 2**32)
            returns = np.random.normal(0.001, 0.015, n)
            prices = np.cumprod(1 + returns) * base
            prices = np.clip(prices, 100, None)
            vol = np.random.randint(100_000, 600_000, n)

            return pd.DataFrame({
                'Date': dates,
                'Open': prices * np.random.uniform(0.98, 1.02, n),
                'High': prices * np.random.uniform(1.00, 1.05, n),
                'Low':  prices * np.random.uniform(0.95, 1.00, n),
                'Close': prices,
                'Volume': vol
            })
        except Exception as e:
            logger.error("❌ خطا در sample_data: %s", e)
            return None

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """پاک‌سازی، یکسان‌سازی ستون‌ها و افزوده کردن تاریخ شمسی"""
        try:
            if df is None or df.empty:
                return None

            df2 = df.copy()
            # ست کردن ایندکس تاریخ
            if 'Date' in df2.columns:
                df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
                df2 = df2.dropna(subset=['Date']).set_index('Date')
            else:
                df2.index = pd.to_datetime(df2.index, errors='coerce')
                df2 = df2.dropna()

            # حذف تکراری
            df2 = df2[~df2.index.duplicated(keep='first')]

            # استانداردسازی نام ستون‌ها
            mapping = {}
            for col in df2.columns:
                lc = col.lower()
                if lc in ['open', 'o', 'first', 'yesterday']:
                    mapping[col] = 'Open'
                elif lc in ['high', 'h', 'max']:
                    mapping[col] = 'High'
                elif lc in ['low', 'l', 'min']:
                    mapping[col] = 'Low'
                elif lc in ['close', 'c', 'last', 'final']:
                    mapping[col] = 'Close'
                elif lc in ['volume', 'vol', 'count', 'value', 'v']:
                    mapping[col] = 'Volume'

            df2 = df2.rename(columns=mapping)

            # اطمینان از وجود همه ستون‌های مورد نیاز
            for req in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if req not in df2.columns:
                    df2[req] = 0

            # تبدیل به عددی
            for numcol in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df2[numcol] = pd.to_numeric(df2[numcol], errors='coerce').fillna(0)

            # افزودن تاریخ شمسی
            df2['JalaliDate'] = [
                JalaliDate.to_jalali(d).strftime('%Y/%m/%d') for d in df2.index
            ]

            df2 = df2.sort_index()
            logger.info("✅ داده آماده شد: %d رکورد", len(df2))
            return df2

        except Exception as e:
            logger.error("❌ خطا در prepare_data: %s", e)
            return None

    def get_data_summary(self, df: pd.DataFrame | None) -> dict:
        if df is None or df.empty:
            return {}
        return {
            "shape": df.shape,
            "columns": list(df.columns),
            "describe": df.describe().to_dict()
        }

    def test_data_sources(self, symbols=None, days_list=None) -> dict:
        if symbols is None:
            symbols = ['فولاد', 'خودرو']
        if days_list is None:
            days_list = [50, 100]
        results = {}
        for sym in symbols:
            for d in days_list:
                df = self.get_stock_data(sym, d, online=True)
                results[f"{sym}_{d}"] = df
                time.sleep(1)
        return results


if __name__ == "__main__":
    mgr = TehranStockDataManager()
    df_online = mgr.get_stock_data("فولاد", days=50, online=True)
    logger.info("خلاصه آنلاین: %s", mgr.get_data_summary(df_online))
    df_offline = mgr.get_stock_data("فولاد", days=50, online=False)
    logger.info("خلاصه آفلاین: %s", mgr.get_data_summary(df_offline))
