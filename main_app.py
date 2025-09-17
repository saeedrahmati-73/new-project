import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from data_collector import collect_stock_data
    from price_predictor import PricePredictor
    from data_processor import TSEDataProcessor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required files are in the same directory")

class StockPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.setup_styles()
        self.create_widgets()
        self.predictor = None

    def setup_window(self):
        """تنظیمات پنجره اصلی"""
        self.root.title("Tehran Stock Exchange - Price Predictor 🤖")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')

        # وسط قرار دادن پنجره
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1400 // 2)
        y = (self.root.winfo_screenheight() // 2) - (900 // 2)
        self.root.geometry(f"1400x900+{x}+{y}")

    def setup_styles(self):
        """تنظیم استایل‌های رابط کاربری"""
        # استایل matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = '#f8f9fa'
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 10

        # استایل‌های دکمه‌ها
        self.button_style = {
            'bg': '#4CAF50',
            'fg': 'white',
            'font': ('Arial', 11, 'bold'),
            'relief': 'raised',
            'bd': 2,
            'padx': 15,
            'pady': 5,
            'cursor': 'hand2'
        }

        self.secondary_button_style = {
            'bg': '#2196F3',
            'fg': 'white',
            'font': ('Arial', 10, 'bold'),
            'relief': 'raised',
            'bd': 2,
            'padx': 10,
            'pady': 3,
            'cursor': 'hand2'
        }

        # استایل برچسب‌ها
        self.label_style = {
            'font': ('Arial', 11),
            'bg': '#f0f0f0',
            'fg': '#333333'
        }

        self.title_style = {
            'font': ('Arial', 16, 'bold'),
            'bg': '#f0f0f0',
            'fg': '#1a237e'
        }

    def create_widgets(self):
        """ایجاد عناصر رابط کاربری"""
        # فریم اصلی
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # عنوان اصلی
        title_label = tk.Label(main_frame,
                              text="Tehran Stock Exchange - AI Price Predictor",
                              **self.title_style)
        title_label.pack(pady=(0, 20))

        # فریم کنترل‌ها
        control_frame = tk.Frame(main_frame, bg='#f0f0f0', relief=tk.RAISED, bd=1)
        control_frame.pack(fill=tk.X, pady=(0, 10))

        # ردیف اول کنترل‌ها
        row1_frame = tk.Frame(control_frame, bg='#f0f0f0')
        row1_frame.pack(fill=tk.X, padx=15, pady=10)

        # نماد سهم
        tk.Label(row1_frame, text="Stock Symbol:", **self.label_style).pack(side=tk.LEFT)
        self.symbol_entry = tk.Entry(row1_frame, font=('Arial', 11), width=15)
        self.symbol_entry.pack(side=tk.LEFT, padx=(5, 20))
        self.symbol_entry.insert(0, "فولاد")

        # روزهای پیش‌بینی
        tk.Label(row1_frame, text="Prediction Days:", **self.label_style).pack(side=tk.LEFT)
        self.days_var = tk.StringVar(value="5")
        days_combo = ttk.Combobox(row1_frame, textvariable=self.days_var,
                                 values=["3", "5", "7", "10", "15"],
                                 width=8, font=('Arial', 11))
        days_combo.pack(side=tk.LEFT, padx=(5, 20))

        # دکمه اجرای پیش‌بینی
        predict_btn = tk.Button(row1_frame, text="🔮 Run Prediction",
                               command=self.run_prediction, **self.button_style)
        predict_btn.pack(side=tk.LEFT, padx=(20, 10))

        # دکمه پاک کردن
        clear_btn = tk.Button(row1_frame, text="🗑️ Clear",
                             command=self.clear_results, **self.secondary_button_style)
        clear_btn.pack(side=tk.LEFT)

        # فریم محتوای اصلی
        content_frame = tk.Frame(main_frame, bg='#f0f0f0')
        content_frame.pack(fill=tk.BOTH, expand=True)

        # فریم چپ (نمودار)
        left_frame = tk.Frame(content_frame, bg='#f0f0f0')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # عنوان نمودار
        chart_title = tk.Label(left_frame,
                              text="📈 Price Analysis & Prediction for Future Days",
                              font=('Arial', 14, 'bold'),
                              bg='#f0f0f0', fg='#1976d2')
        chart_title.pack(pady=(0, 10))

        # فریم نمودار
        self.chart_frame = tk.Frame(left_frame, bg='white', relief=tk.SUNKEN, bd=2)
        self.chart_frame.pack(fill=tk.BOTH, expand=True)

        # فریم راست (نتایج)
        right_frame = tk.Frame(content_frame, bg='#f0f0f0', width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        right_frame.pack_propagate(False)

        # عنوان نتایج
        results_title = tk.Label(right_frame,
                                text="📊 Analysis Results",
                                font=('Arial', 14, 'bold'),
                                bg='#f0f0f0', fg='#1976d2')
        results_title.pack(pady=(0, 10))

        # فریم نتایج با اسکرول
        results_frame = tk.Frame(right_frame, bg='white', relief=tk.SUNKEN, bd=2)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5)

        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            font=('Consolas', 10),
            bg='#ffffff',
            fg='#333333',
            selectbackground='#e3f2fd',
            wrap=tk.WORD
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # نوار وضعیت
        status_frame = tk.Frame(main_frame, bg='#f0f0f0')
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Enter a stock symbol and click 'Run Prediction'")

        self.status_bar = tk.Label(status_frame,
                                  textvariable=self.status_var,
                                  relief=tk.SUNKEN,
                                  anchor=tk.W,
                                  font=('Arial', 10),
                                  bg='#e0e0e0',
                                  fg='#666666')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # پیام خوشامدگویی
        self.show_welcome_message()

    def show_welcome_message(self):
        """نمایش پیام خوشامدگویی"""
        welcome_text = """
🚀 به پیش‌بینی‌گر هوشمند قیمت سهام خوش آمدید 🚀

📋 راهنما:
• نماد سهم را وارد کنید (مثال: فولاد، شستا)
• تعداد روزهای پیش‌بینی را انتخاب کنید (۳ تا ۱۵ روز)
• برای شروع تحلیل، روی دکمه "اجرای پیش‌بینی" کلیک کنید
• نتایج را در این پنل و نمودارها را در سمت چپ مشاهده کنید

🔧 ویژگی‌ها:
• دریافت آنی داده‌ها از بورس تهران
• الگوریتم‌های پیشرفته هوش مصنوعی
• نمودارهای تعاملی قیمت
• شاخص‌های تحلیل تکنیکال
• پیش‌بینی قیمت آینده

💡 نکات:
• از نمادهای فارسی استفاده کنید
• روزهای پیش‌بینی بیشتر = تحلیل روند گسترده‌تر
• برای تحلیل بهتر، بازه‌های زمانی مختلف را بررسی کنید

آماده برای پیش‌بینی آینده قیمت‌ها! 📈
        """

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, welcome_text)
        self.results_text.configure(state='disabled')

    def update_status(self, message):
        """بروزرسانی نوار وضعیت"""
        self.status_var.set(message)
        self.root.update_idletasks()

    def run_prediction(self):
        """اجرای پیش‌بینی قیمت"""
        symbol = self.symbol_entry.get().strip()
        if not symbol:
            messagebox.showerror("خطا", "لطفا نماد سهم را وارد کنید!")
            return

        try:
            prediction_days = int(self.days_var.get())
        except ValueError:
            messagebox.showerror("خطا", "لطفا تعداد روز پیش‌بینی را به درستی انتخاب کنید!")
            return

        self.update_status("🔄 در حال دریافت داده‌های سهام...")

        try:
            # 1. جمع‌آوری داده‌ها
            raw_data = collect_stock_data(symbol, days=100) # Get more data for processing
            if raw_data is None or raw_data.empty:
                raise Exception("داده‌ای برای این نماد یافت نشد")

            self.update_status("⚙️ در حال پردازش داده‌ها و محاسبه شاخص‌ها...")

            # 2. پردازش داده‌ها
            processor = TSEDataProcessor()
            processed_data = processor.process_historical_data(raw_data.to_dict('records'), symbol)
            if processed_data is None or processed_data.empty:
                raise Exception("پردازش داده‌ها با مشکل مواجه شد")

            self.update_status("🤖 در حال آموزش مدل و پیش‌بینی...")

            # 3. ایجاد و آموزش مدل و پیش‌بینی
            self.predictor = PricePredictor()
            # The new predictor class handles everything in one method
            results = self.predictor.predict_prices(processed_data, symbol)

            if not results or 'future_predictions' not in results:
                raise Exception("مدل پیش‌بینی موفق به تولید نتیجه نشد")

            predictions = results['future_predictions']

            # 4. نمایش نتایج
            # The display_results function needs the original (unprocessed) dataframe
            # Let's use the processed one for consistency in plotting
            self.display_results(symbol, processed_data, predictions, prediction_days)
            self.plot_results(symbol, processed_data, predictions)

            self.update_status(f"✅ پیش‌بینی برای {symbol} در {prediction_days} روز آینده تکمیل شد")

        except Exception as e:
            error_msg = f"خطا: {str(e)}"
            messagebox.showerror("خطای پیش‌بینی", error_msg)
            self.update_status(f"❌ خطا: {str(e)}")
            import traceback
            traceback.print_exc()

    def display_results(self, symbol, df, predictions, days):
        """نمایش نتایج در پنل راست"""
        self.results_text.configure(state='normal')
        self.results_text.delete(1.0, tk.END)

        # آمار کلی
        current_price = df['Close'].iloc[-1]
        avg_prediction = np.mean(predictions)
        price_change = avg_prediction - current_price
        change_percent = (price_change / current_price) * 100

        # تحلیل روند
        if change_percent > 2:
            trend = "📈 صعودی"
        elif change_percent < -2:
            trend = "📉 نزولی"
        else:
            trend = "➡️ خنثی"

        results_text = f"""
🏢 گزارش تحلیل سهام
{'='*40}

📊 نماد: {symbol}
📅 تاریخ تحلیل: {datetime.now().strftime('%Y-%m-%d %H:%M')}
🔮 دوره پیش‌بینی: {days} روز

💰 داده‌های فعلی بازار
{'='*40}
قیمت فعلی: {current_price:,.0f} ریال
روند بازار: {trend}
میانگین قیمت پیش‌بینی شده: {avg_prediction:,.0f} ریال
تغییر قیمت: {price_change:+,.0f} ریال
درصد تغییر: {change_percent:+.2f}%

📈 پیش‌بینی روزانه
{'='*40}
"""

        # پیش‌بینی‌های روزانه
        for i, pred in enumerate(predictions, 1):
            date = datetime.now() + timedelta(days=i)
            daily_change = ((pred - current_price) / current_price) * 100
            trend_icon = "📈" if daily_change > 0 else "📉" if daily_change < 0 else "➡️"

            results_text += f"روز {i} ({date.strftime('%m/%d')}): {pred:,.0f} ریال {trend_icon} ({daily_change:+.1f}%)\n"

        # تحلیل تکنیکال
        results_text += f"""

🔍 تحلیل تکنیکال
{'='*40}
تعداد روزهای تحلیل شده: {len(df)} روز
نوسان قیمت: {df['Close'].std():.0f} ریال
بیشترین قیمت (۳۰ روز): {df['High'].tail(30).max():,.0f} ریال
کمترین قیمت (۳۰ روز): {df['Low'].tail(30).min():,.0f} ریال

📊 ارزیابی ریسک
{'='*40}
"""

        # ارزیابی ریسک
        volatility = df['Close'].pct_change().std() * 100
        if volatility > 3:
            risk_level = "🔴 ریسک بالا"
        elif volatility > 1.5:
            risk_level = "🟡 ریسک متوسط"
        else:
            risk_level = "🟢 ریسک پایین"

        results_text += f"سطح ریسک: {risk_level}\n"
        results_text += f"نوسان قیمت: {volatility:.2f}%\n"

        # توصیه‌های سرمایه‌گذاری
        results_text += f"""

💡 پیشنهاد سرمایه‌گذاری
{'='*40}
"""

        if change_percent > 5:
            recommendation = "🚀 خرید قوی - انتظار روند صعودی قابل توجه"
        elif change_percent > 2:
            recommendation = "📈 خرید - پیش‌بینی روند مثبت"
        elif change_percent > -2:
            recommendation = "⏳ نگهداری - انتظار قیمت پایدار"
        elif change_percent > -5:
            recommendation = "📉 فروش - احتمال روند نزولی"
        else:
            recommendation = "🚨 فروش قوی - انتظار کاهش قابل توجه"

        results_text += f"{recommendation}\n"

        results_text += f"""

⚠️ سلب مسئولیت
{'='*40}
این تحلیل بر اساس پیش‌بینی‌های هوش مصنوعی و
داده‌های تاریخی است. قیمت سهام تابع ریسک‌ها
و نوسانات بازار است. همیشه قبل از تصمیم‌گیری
برای سرمایه‌گذاری با مشاوران مالی مشورت کنید.

تولید شده توسط پیش‌بینی‌گر هوشمند بورس تهران 🤖
"""

        self.results_text.insert(tk.END, results_text)
        self.results_text.configure(state='disabled')

    def plot_results(self, symbol, df, predictions):
        """رسم نمودارها"""
        # پاک کردن نمودار قبلی
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # ایجاد نمودار
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle(f'Stock Analysis: {symbol}', fontsize=16, fontweight='bold')

        # نمودار قیمت تاریخی
        dates = pd.to_datetime(df.index)
        ax1.plot(dates, df['Close'], label='Historical Price', color='#1f77b4', linewidth=2)
        ax1.fill_between(dates, df['Low'], df['High'], alpha=0.3, color='#1f77b4')

        # نمودار پیش‌بینی
        future_dates = [datetime.now() + timedelta(days=i) for i in range(1, len(predictions) + 1)]
        current_price = df['Close'].iloc[-1]

        # خط اتصال
        ax1.plot([dates.iloc[-1], future_dates[0]], [current_price, predictions[0]],
                'r--', alpha=0.7, linewidth=1)

        ax1.plot(future_dates, predictions, 'ro-', label='Predicted Price',
                color='#ff4444', linewidth=2, markersize=6)

        ax1.set_title('Price History & Predictions', fontweight='bold')
        ax1.set_ylabel('Price (Rials)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # نمودار تغییرات درصدی
        price_changes = df['Close'].pct_change() * 100
        ax2.bar(dates[1:], price_changes[1:], alpha=0.6,
               color=['green' if x > 0 else 'red' for x in price_changes[1:]])

        # تغییرات پیش‌بینی شده
        pred_changes = [(pred - current_price) / current_price * 100 for pred in predictions]
        ax2.bar(future_dates, pred_changes, alpha=0.8,
               color=['darkgreen' if x > 0 else 'darkred' for x in pred_changes])

        ax2.set_title('Daily Price Changes (%)', fontweight='bold')
        ax2.set_ylabel('Change (%)', fontweight='bold')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # اضافه کردن نمودار به GUI
        canvas = FigureCanvasTkAgg(fig, self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def clear_results(self):
        """پاک کردن نتایج"""
        # پاک کردن نمودار
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # پاک کردن متن نتایج
        self.show_welcome_message()

        # ریست کردن نوار وضعیت
        self.update_status("Ready - Enter a stock symbol and click 'Run Prediction'")

        # پاک کردن مدل
        self.predictor = None

def main():
    """تابع اصلی برنامه"""
    try:
        root = tk.Tk()
        app = StockPredictorGUI(root)

        # اضافه کردن منوی کیبورد
        def on_key_press(event):
            if event.keysym == 'Return':
                app.run_prediction()
            elif event.keysym == 'Escape':
                app.clear_results()

        root.bind('<Key>', on_key_press)
        root.focus_set()

        # اجرای برنامه
        root.mainloop()

    except Exception as e:
        print(f"Error starting application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application:\n{e}")

if __name__ == "__main__":
    main()
