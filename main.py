import argparse, itertools, os, time, warnings, webbrowser
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ta
import yfinance as yf
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from bokeh.io import output_file, save
from bokeh.resources import INLINE
from bokeh.layouts import column
from bokeh.plotting import figure
from bokeh.models import (
    BoxAnnotation, ColumnDataSource, CrosshairTool, DataTable,
    DatetimeTickFormatter, Div, HoverTool, LinearAxis, Range1d,
    RangeSlider, RangeTool, TabPanel, TableColumn, Tabs
)
from bokeh.models import tickers
from bokeh.palettes import Dark2_5 as palette
def log_exponential_func(x, lna, b, c):
    return lna + b * x + c * np.log(x + 1e-6)

def exponential_func(x, a, b, c):
    return a * np.exp(b * x) * (x + 1e-6)**c


def gen_all_time(file):
    # Read CSV directly into DataFrame
    try:
        df = pd.read_csv(file, parse_dates=['DateTime'], usecols=['DateTime', 'Series 1 (y)'])
    except ValueError:
        # Fallback: read all and rename common variants
        df = pd.read_csv(file, parse_dates=True, infer_datetime_format=True)
        date_col = next((c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()), df.columns[0])
        val_col = next((c for c in df.columns if c.lower() in ['close', 'price', 'value', 'y']), df.columns[1])
        df = df[[date_col, val_col]].rename(columns={date_col: 'DateTime', val_col: value_column})
        
        
    df = df[df['DateTime'].notna()].copy()
    df['Date'] = df['DateTime'].dt.date
    df['y'] = df.iloc[:, 1].astype(float)

    # Daily close (last value of each day)
    daily = df.groupby(df['DateTime'].dt.date)['y'].last().reset_index()
    daily['DateTime'] = pd.to_datetime(daily['DateTime'])

    # Core indicators
    daily['daily_pct'] = daily['y'].pct_change().fillna(0) * 100

    for w in [5, 20, 60]:
        daily[f'ma_{w}'] = daily['y'].rolling(w).mean()
        daily[f'ma_pct_{w}'] = daily['daily_pct'].rolling(w).mean()
    daily['ma_pct_250'] = daily['daily_pct'].rolling(250).mean()



    # exponential moving average (EMA)
    for w in [5, 20, 60]:
        daily[f'ema_{w}'] = daily['y'].ewm(span=w, adjust=False).mean()
        daily[f'ema_pct_{w}'] = daily['daily_pct'].ewm(span=w, adjust=False).mean()

    daily['ema_pct_250'] = daily['daily_pct'].ewm(span=250, adjust=False).mean()



    # Differences between MAs
    daily['avg_5_20'] = daily['ma_5'] - daily['ma_20']
    daily['avg_5_60'] = daily['ma_5'] - daily['ma_60']
    daily['avg_20_60'] = daily['ma_20'] - daily['ma_60']

    # RSI
    delta = daily['y'].diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    daily['rsi'] = 100 - 100 / (1 + up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean())

    # Fix future date
    today = datetime.now().date()
    if daily['DateTime'].iloc[-1].date() > today:
        daily.loc[daily.index[-1], 'DateTime'] = pd.Timestamp(today)

    return daily


def gen_each_year(all_time):
    df = pd.DataFrame(all_time)
    df['year'] = pd.to_datetime(df['DateTime']).dt.year
    
    grouped = df.groupby('year')
    
    each_year = {
        year: {
            'DateTime': group['DateTime'].tolist(),
            'y': group['y'].tolist(),
            'average_y_5': group['ma_5'].tolist(),
            'average_y_20': group['ma_20'].tolist(),
            'average_y_60': group['ma_60'].tolist(),
            'average_y_5_20': group['avg_5_20'].tolist(),
            'average_y_5_60': group['avg_5_60'].tolist(),
            'average_y_20_60': group['avg_20_60'].tolist(),
            'rsi': group['rsi'].tolist(),
            'daily_percent_increase': group['daily_pct'].tolist(),
            'daily_percent_increase_ma_5': group['ma_pct_5'].tolist(),
            'daily_percent_increase_ma_20': group['ma_pct_20'].tolist(),
            'daily_percent_increase_ma_60': group['ma_pct_60'].tolist(),
            'daily_percent_increase_ma_250': group['ma_pct_250'].tolist(),
        }
        for year, group in grouped
    }
    
    return each_year


def gen_year_over_year(all_time_df):
    df = all_time_df.copy()
    df['year'] = df['DateTime'].dt.year
    df['day_offset'] = df.groupby('year').cumcount()
    df['start_val'] = df['y'].where(df['day_offset'] == 0).groupby(df['year']).ffill()
    
    df['y_diff'] = df['y'] - df['start_val']
    df['pct_inc'] = (df['y_diff'] / df['start_val'] * 100).round(3)
    
    grouped = df.groupby('year')
    return {
        year: {
            'DateTime': group['day_offset'].tolist(),
            'y_values': group['y_diff'].tolist(),
            'percent_increase': group['pct_inc'].tolist(),
        }
        for year, group in grouped
    }


def gen_all_time_high(df):
    y = df['y'].values
    dates = df['DateTime'].values

    # Find local maxima (peaks) and minima (troughs)
    peaks_idx = argrelextrema(y, np.greater, order=5)[0]   # 5-day window to avoid noise
    troughs_idx = argrelextrema(y, np.less, order=5)[0]

    highs = []
    trough_iter = iter(troughs_idx)
    next_trough_idx = next(trough_iter, None)

    for peak_idx in peaks_idx:
        peak_val = y[peak_idx]
        peak_date = dates[peak_idx]

        # Find first trough after this peak
        while next_trough_idx is not None and next_trough_idx <= peak_idx:
            next_trough_idx = next(trough_iter, None)
        if next_trough_idx is None:
            break

        trough_val = y[next_trough_idx]
        trough_date = dates[next_trough_idx]

        pct_drop = (peak_val - trough_val) / peak_val * 100
        days = (pd.to_datetime(trough_date) - pd.to_datetime(peak_date)).days

        if pct_drop >= 0.5 and days >= 3:  # meaningful correction
            highs.append({
                'date': pd.to_datetime(peak_date).date(),
                'cycle_high': round(peak_val, 3),
                'cycle_low': round(trough_val, 3),
                'difference': round(peak_val - trough_val, 3),
                'days': days,
                'percent': round(-pct_drop, 3),
            })

    return highs


def gen_all_time_downside(df):
    y = df['y']
    cycle_high = y.cummax()
    downside = (cycle_high - y).round(3)
    return df.assign(
        downside=-downside,
        downside_percent=((y / cycle_high - 1) * 100).round(3)
    )


def gen_best_fit(data_set):
    df = pd.DataFrame({
        'x': pd.to_datetime(data_set['DateTime']),
        'y': data_set['y']
    })
    df['t'] = np.arange(len(df))  # time index

    y = df['y']
    t = df['t']

    # Linear fit (slope, R²)
    slope, intercept = np.polyfit(t, y, 1)
    y_pred = slope * t + intercept
    ssr = np.sum((y - y_pred)**2)
    sst = np.sum((y - y.mean())**2)
    r2 = 1 - ssr / sst

    # Quadratic fit
    quad_coefs = np.polyfit(t, y, 2)
    quad_fit = np.polyval(quad_coefs, t)

    # Percent increases
    pct_total = (y.iloc[-1] / y.iloc[0] - 1) * 100
    pct_daily_avg = pct_total / len(y)

    # Exponential/log fit
    y_safe = np.maximum(y - 1, 1e-10)
    try:
        params, _ = curve_fit(log_exponential_func, t, np.log(y_safe), p0=[1, 0.01, 1], maxfev=5000)
        lna, b, c = params
        exp_fit = exponential_func(t, np.exp(lna), b, c)
    except:
        exp_fit = [np.nan] * len(y)

    return {
        'slope': round(slope, 3),
        'r2': round(r2, 3),
        'percent_increase': round(pct_total, 3),
        'percent_increase_daily_avg': round(pct_daily_avg, 3),
        'best_fit_line': y_pred.round(3).tolist(),
        'theta_fit_list_2': quad_fit.round(3).tolist(),
        'best_fit_exp': exp_fit.round(3).tolist(),
    }


def gen_benchmark(data_set_df):
    tickers = ["^GSPC", "^IXIC", "^DJI"]
    dates = data_set_df['DateTime'][-20:].tolist()
    y_values = data_set_df['y'][-20:].tolist()
    benchmark = {'portfolio': [], 'DateTime': list(range(len(dates)))}


    try:
        data = yf.download(tickers, start=dates[0], auto_adjust=True, progress=False)
        if data.empty:
            raise ValueError("No data returned.")
        
        new_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
        data = data.reindex(new_index).interpolate(method='time')
        
        start_port = y_values[0]
        benchmark['portfolio'] = [round((v - start_port) / start_port * 100, 3) for v in y_values]
        
        for ticker in tickers:
            start_val = data.loc[dates[0], ('Close', ticker)]
            benchmark[ticker] = [round((data.loc[d, ('Close', ticker)] - start_val) / start_val * 100, 3) for d in dates]

    except Exception as e:
        print(f"Error fetching benchmark data: {e}")
        print("Falling back to default benchmark values.")
        benchmark['portfolio'] = [0.0] * len(dates)
        for ticker in tickers:
            benchmark[ticker] = [0.0] * len(dates)

    return benchmark


def gen_forecast(all_time_df, forecast_steps=252):
    y_values = all_time_df['y'].values
    dates = all_time_df['DateTime'].values

    # Find first day of current year
    current_year = pd.to_datetime(dates[-1]).year
    current_year_start_idx = next(
        (i for i in reversed(range(len(dates))) if pd.to_datetime(dates[i]).year < current_year),
        -1
    ) + 1

    y_past = y_values[:current_year_start_idx]

    # Models on full data
    model_add = ExponentialSmoothing(y_values, seasonal_periods=252, trend='add', seasonal='add', use_boxcox=True, damped_trend=True).fit()
    model_mul = ExponentialSmoothing(y_values, seasonal_periods=252, trend='mul', seasonal='add', use_boxcox=True, damped_trend=True).fit()

    # Models excluding current year
    past_add = ExponentialSmoothing(y_past, seasonal_periods=252, trend='add', seasonal='add', use_boxcox=True, damped_trend=True).fit()
    past_mul = ExponentialSmoothing(y_past, seasonal_periods=252, trend='mul', seasonal='add', use_boxcox=True, damped_trend=True).fit()

    # Forecasts
    fc_add = model_add.forecast(forecast_steps).tolist()
    fc_mul = model_mul.forecast(forecast_steps).tolist()
    fc_past_add = past_add.forecast(forecast_steps).tolist()
    fc_past_mul = past_mul.forecast(forecast_steps).tolist()

    fc_avg = [(a + m) / 2 for a, m in zip(fc_add, fc_mul)]
    fc_past_avg = [(a + m) / 2 for a, m in zip(fc_past_add, fc_past_mul)]

    # Proper business-day dates
    last_date = pd.to_datetime(dates[-1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B').tolist()

    this_year_start_date = pd.to_datetime(dates[current_year_start_idx - 1] if current_year_start_idx > 0 else dates[0])
    future_dates_this_year = pd.bdate_range(start=this_year_start_date + pd.Timedelta(days=1), periods=forecast_steps, freq='B').tolist()

    return {
        'x_values': future_dates,
        'x_values_this_year': future_dates_this_year,
        'y_values_add': fc_add,
        'y_values_mul': fc_mul,
        'y_values_avg': fc_avg,
        'y_values_this_year_add': fc_past_add,
        'y_values_this_year_mul': fc_past_mul,
        'y_values_this_year_avg': fc_past_avg,
    }


def gen_time_frame_stats(all_time):
    y = all_time['y'].values
    stats = {}
    
    for days in (5, 20, 60, 250):
        period = -int(days)
        start, end = y[period], y[-1]
        total_change = (end - start) / start * 100
        
        # Linear fit for slope & R²
        x = np.arange(len(y[period:]))
        slope, intercept = np.polyfit(x, y[period:], 1)
        y_pred = slope * x + intercept
        ssr = np.sum((y[period:] - y_pred) ** 2)
        sst = np.sum((y[period:] - y[period:].mean()) ** 2)
        r2 = 1 - ssr / sst if sst != 0 else 0
        
        stats[days] = {
            'percent_increase': round(total_change, 3),
            'daily_percent_increase': round(total_change / days, 3),
            'slope': round(slope, 3),
            'r2': round(r2, 3)
        }
    
    return stats


def gen_bokeh_forecast_chart(data_set, forecasts):

    tabs = []

    p_forecast_this_year = figure(
        x_axis_type="datetime",
        sizing_mode="scale_width",
        aspect_ratio=11/5,
        title="Value",
        x_axis_label="x",
        y_axis_label="y",
        )

    source = ColumnDataSource(data={
        'x': data_set['DateTime'],
        'y_values': data_set['y'],
    })

    forecast_source = ColumnDataSource(data={
        'x_values_this_year': forecasts['x_values_this_year'],
        'y_values_this_year_add': forecasts['y_values_this_year_add'],
        'y_values_this_year_mul': forecasts['y_values_this_year_mul'],
        'y_values_this_year_avg': forecasts['y_values_this_year_avg'],
        'x_values': forecasts['x_values'],
        'y_values_add': forecasts['y_values_add'],
        'y_values_mul': forecasts['y_values_mul'],
        'y_values_avg': forecasts['y_values_avg'],
    })

    # add multiple renderers
    line1 = p_forecast_this_year.line(
        'x',
        'y_values',
        source=source,
        legend_label="Value",
        color="blue",
        line_width=1,
    )

    p_forecast_this_year.line(
        'x_values_this_year',
        'y_values_this_year_add',
        source=forecast_source,
        legend_label="Holt-Winters add",
        color="orange",
        line_width=1,
    )
    p_forecast_this_year.line(
        'x_values_this_year',
        'y_values_this_year_mul',
        source=forecast_source,
        legend_label="Holt-Winters mul",
        color="red",
        line_width=1,
    )
    p_forecast_this_year.line(
        'x_values_this_year',
        'y_values_this_year_avg',
        source=forecast_source,
        legend_label="Holt-Winters avg",
        color="green",
        line_width=1,
    )

    hover = HoverTool(
        tooltips=[
            ("Date", "@x{%F}"),
            ("Tool", "$y{0,0.00}"),
            ("Value", "@y_values{0,0.00}"),
            ("add", "@y_values_this_year_add{0,0.00}"),
            ("mul", "@y_values_this_year_mul{0,0.00}"),
            ("avg", "@y_values_this_year_avg{0,0.00}"),

        ],
        formatters={'@x': 'datetime'},
        mode='vline',
        renderers=[line1]
    )
    cross = CrosshairTool()
    p_forecast_this_year.add_tools(hover, cross)

    p_forecast_this_year.xaxis[0].formatter = DatetimeTickFormatter(
        days="%m - %Y",
        months="%m - %Y",
    )
    p_forecast_this_year.xaxis.ticker = tickers.MonthsTicker(
        months=list(range(0, 13, 1))
    )
    p_forecast_this_year.xaxis.major_label_orientation = 0.9

    p_forecast_this_year.legend.click_policy = "hide"
    p_forecast_this_year.legend.location = "top_left"

    p_forecast = figure(
        x_axis_type="datetime",
        sizing_mode="scale_width",
        aspect_ratio=11/5,
        title="Value",
        x_axis_label="x",
        y_axis_label="y",
        )

    # add multiple renderers
    p_forecast.line(
        'x',
        'y_values',
        source=source,
        legend_label="Value",
        color="blue",
        line_width=1,
    )

    p_forecast.line(
        forecasts['x_values'],
        forecasts['y_values_add'],
        legend_label="Holt-Winters add",
        color="orange",
        line_width=1,
    )
    p_forecast.line(
        forecasts['x_values'],
        forecasts['y_values_mul'],
        legend_label="Holt-Winters mul",
        color="red",
        line_width=1,
    )
    p_forecast.line(
        forecasts['x_values'],
        forecasts['y_values_avg'],
        legend_label="Holt-Winters avg",
        color="green",
        line_width=1,
    )

    p_forecast.xaxis[0].formatter = DatetimeTickFormatter(
        days="%m - %Y",
        months="%m - %Y",
    )
    p_forecast.xaxis.ticker = tickers.MonthsTicker(
        months=list(range(0, 13, 1))
    )
    p_forecast.xaxis.major_label_orientation = 0.9

    p_forecast.legend.click_policy = "hide"
    p_forecast.legend.location = "top_left"

    tabs.append(TabPanel(
        child=column(
            p_forecast_this_year,
            p_forecast,
            sizing_mode="scale_width",
        ),
        title="Forecast",
    ))

    return tabs


def gen_bokeh_chart(data_set_id, data_set, each_year, time_frame, benchmarks, all_time_high):
    chart_width = 1500
    chart_height = 800
    tabs = []

    date_buffer = int((data_set['DateTime'][-1] - data_set['DateTime'][0]).days * 0.025)

    source = ColumnDataSource(data={
        'x': data_set['DateTime'],
        'y_value': data_set['y'],
        'y_sma5': data_set['ma_5'],
        'y_sma20': data_set['ma_20'],
        'y_sma60': data_set['ma_60'],
        'y_ema5': data_set['ema_5'],
        'y_ema20': data_set['ema_20'],
        'y_ema60': data_set['ema_60'],
        'y_best_fit_line': data_set['best_fit_line'],
        'y_theta_fit_list_2': data_set['theta_fit_list_2'],
        'y_best_fit_exp': data_set['best_fit_exp'],
    })

    p_all = figure(
        x_axis_type="datetime",
        sizing_mode="scale_width",
        aspect_ratio=11/5,
        title="Value",
        x_axis_label="x",
        y_axis_label="y",
        x_range=(
            data_set['DateTime'][0] - timedelta(days=date_buffer),
            data_set['DateTime'][-1] + timedelta(days=date_buffer),
        )
    )

    line1 = p_all.line('x', 'y_value', source=source, legend_label="Value", color="blue", line_width=1)
    p_all.line('x', 'y_sma5', source=source, legend_label="SMA 5", color="orange", line_width=1, line_dash="dashed")
    p_all.line('x', 'y_sma20', source=source, legend_label="SMA 20", color="green", line_width=1, line_dash="dashed")
    p_all.line('x', 'y_sma60', source=source, legend_label="SMA 60", color="red", line_width=1, line_dash="dashed")
    p_all.line('x', 'y_ema5', source=source, legend_label="EMA 5", color="orange", line_width=1, line_dash="dotted")
    p_all.line('x', 'y_ema20', source=source, legend_label="EMA 20", color="green", line_width=1, line_dash="dotted")
    p_all.line('x', 'y_ema60', source=source, legend_label="EMA 60", color="red", line_width=1, line_dash="dotted")
    p_all.line('x', 'y_best_fit_line', source=source, legend_label="Linear", color="purple", line_width=1)
    p_all.line('x', 'y_theta_fit_list_2', source=source, legend_label="Polynomial", color="gold", line_width=1)
    p_all.line('x', 'y_best_fit_exp', source=source, legend_label="Exponential", color="brown", line_width=1)

    hover = HoverTool(
        tooltips=[
            ("Date", "@x{%F}"),
            ("Tool", "$y{0,0.00}"),
            ("Value", "@y_value{0,0.00}"),
            ("SMA 5", "@y_sma5{0,0.00}"),
            ("SMA 20", "@y_sma20{0,0.00}"),
            ("SMA 60", "@y_sma60{0,0.00}"),
            ("Linear", "@y_best_fit_line{0,0.00}"),
            ("Polynomial", "@y_theta_fit_list_2{0,0.00}"),
            ("Exponential", "@y_best_fit_exp{0,0.00}"),
        ],
        formatters={'@x': 'datetime'},
        mode='vline',
        renderers=[line1]
    )
    cross = CrosshairTool()
    p_all.add_tools(hover, cross)

    p_all.xaxis[0].formatter = DatetimeTickFormatter(days="%m - %Y", months="%m - %Y")
    p_all.xaxis.ticker = tickers.MonthsTicker(months=list(range(0, 13, 1)))
    p_all.xaxis.major_label_orientation = 0.9
    p_all.legend.click_policy = "hide"
    p_all.legend.location = "top_left"

    range_slider = RangeSlider(
        title="Adjust y-axis range",
        start=0,
        end=max(data_set['y']) * 1.15,
        step=1,
        value=(0, max(data_set['y']) * 1.15),
        margin=(5, 80, 5, 80),
        sizing_mode="stretch_width"
    )
    range_slider.js_link("value", p_all.y_range, "start", attr_selector=0)
    range_slider.js_link("value", p_all.y_range, "end", attr_selector=1)

    p_all_daily_percent_increase = figure(
        x_axis_type="datetime",
        sizing_mode="scale_width",
        aspect_ratio=8,
        title="Daily Percent Increase",
        y_axis_label="y",
        x_range=p_all.x_range,
        y_range=(-5, 5)
    )

    p_all_daily_percent_increase.line(data_set['DateTime'], data_set['daily_pct'], legend_label="Value", color="blue", line_width=1)
    p_all_daily_percent_increase.line(data_set['DateTime'], data_set['ma_pct_5'], legend_label="SMA 5", color="orange", line_width=1)
    p_all_daily_percent_increase.line(data_set['DateTime'], data_set['ma_pct_20'], legend_label="SMA 20", color="green", line_width=1)
    p_all_daily_percent_increase.line(data_set['DateTime'], data_set['ma_pct_60'], legend_label="SMA 60", color="red", line_width=1)
    p_all_daily_percent_increase.line(data_set['DateTime'], data_set['ma_pct_250'], legend_label="SMA 250", color="purple", line_width=1)

    p_all_daily_percent_increase.xaxis.ticker = tickers.MonthsTicker(months=list(range(0, 13, 1)))
    p_all_daily_percent_increase.xaxis.major_label_text_font_size = '0pt'
    p_all_daily_percent_increase.legend.click_policy = "hide"
    p_all_daily_percent_increase.legend.location = "top_left"

    p_all_rsi = figure(
        x_axis_type="datetime",
        sizing_mode="scale_width",
        aspect_ratio=8,
        title="RSI",
        y_axis_label="y",
        x_range=p_all.x_range,
        y_range=(0, 100),
    )

    p_all_rsi.line(data_set['DateTime'], data_set['rsi'], legend_label="RSI", color="blue", line_width=1)
    p_all_rsi.add_layout(BoxAnnotation(top=40, fill_alpha=0.1, fill_color='green', line_color='green'))
    p_all_rsi.add_layout(BoxAnnotation(bottom=80, fill_alpha=0.1, fill_color='red', line_color='red'))

    p_all_rsi.xaxis.ticker = tickers.MonthsTicker(months=list(range(0, 13, 1)))
    p_all_rsi.xaxis.major_label_text_font_size = '0pt'
    p_all_rsi.legend.click_policy = "hide"
    p_all_rsi.legend.location = "top_left"

    p_macd = figure(
        x_axis_type="datetime",
        sizing_mode="scale_width",
        aspect_ratio=6,
        title="MACD",
        x_axis_label="x",
        y_axis_label="y",
        x_range=p_all.x_range,
    )

    p_macd.line(data_set['DateTime'], data_set['avg_5_20'], legend_label="Running average 5 - 20", color="green", line_width=1)
    p_macd.line(data_set['DateTime'], data_set['avg_5_60'], legend_label="Running average 5 - 60", color="orange", line_width=1)
    p_macd.line(data_set['DateTime'], data_set['avg_20_60'], legend_label="Running average 20 - 60", color="red", line_width=1)

    p_macd.xaxis[0].formatter = DatetimeTickFormatter(days="%m - %Y", months="%m - %Y")
    p_macd.xaxis.ticker = tickers.MonthsTicker(months=list(range(0, 13, 1)))
    p_macd.xaxis.major_label_orientation = 0.9
    p_macd.legend.click_policy = "hide"
    p_macd.legend.location = "top_left"

    p_downside = figure(
        x_axis_type="datetime",
        sizing_mode="scale_width",
        aspect_ratio=8,
        title="Downside",
        y_axis_label="y",
        x_range=p_all.x_range,
        y_range=(-2500, 100)
    )

    p_downside.extra_y_ranges = {"foo": Range1d(start=-25, end=1)}
    p_downside.add_layout(LinearAxis(y_range_name="foo"), 'right')

    p_downside.line(data_set['DateTime'], data_set['downside'], legend_label="Downside", color="blue", line_width=1)
    p_downside.line(data_set['DateTime'], data_set['downside_percent'], legend_label="Downside Percent", color="green", line_width=1, y_range_name="foo")

    p_downside.xaxis[0].formatter = DatetimeTickFormatter(days="%m - %Y", months="%m - %Y")
    p_downside.xaxis.ticker = tickers.MonthsTicker(months=list(range(0, 13, 1)))
    p_downside.xaxis.major_label_text_font_size = '0pt'
    p_downside.legend.click_policy = "hide"
    p_downside.legend.location = "bottom_left"

    select = figure(
        title="Range Selecter",
        height=130,
        x_axis_type="datetime",
        y_axis_type=None,
        tools="",
        toolbar_location=None,
        background_fill_color="#efefef",
        sizing_mode="stretch_width",
        margin=(20, 5, 20, 5),
    )

    range_tool = RangeTool(x_range=p_all.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.line(data_set['DateTime'], data_set['y'])
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)

    each_year_stats_title = Div(text="<h4>Yearly Stats</h4>", margin=(5, 80, 5, 80))

    each_year_stats = {
        'year': [],
        'slope': [],
        'r2': [],
        'percent_increase': [],
        'percent_increase_daily_avg': [],
        'total_days': []
    }

    for year in each_year:
        each_year_stats['year'].append(year)
        each_year_stats['slope'].append(each_year[year]['slope'])
        each_year_stats['r2'].append(each_year[year]['r2'])
        each_year_stats['percent_increase'].append(each_year[year]['percent_increase'])
        each_year_stats['percent_increase_daily_avg'].append(each_year[year]['percent_increase_daily_avg'])
        each_year_stats['total_days'].append(len(each_year[year]['y']))

    source = ColumnDataSource(each_year_stats)

    each_year_stats_columns = [
        TableColumn(field="year", title="Date"),
        TableColumn(field="slope", title="slope"),
        TableColumn(field="r2", title="r2"),
        TableColumn(field="percent_increase", title="percent_increase"),
        TableColumn(field="percent_increase_daily_avg", title="percent_increase_daily_avg"),
        TableColumn(field="total_days", title="total_days"),
    ]

    each_year_stats_data_table = DataTable(
        source=source,
        columns=each_year_stats_columns,
        sizing_mode="stretch_width",
        height=280,
        index_position=None,
        margin=(-5, 80, -120, 80),
    )

    time_frame_stats_title = Div(text="<h4>Time Frame Stats</h4>", margin=(5, 80, 5, 80))

    time_frame_stats = {
        'time_frame': [],
        'slope': [],
        'r2': [],
        'percent_increase': [],
        'daily_percent_increase': []
    }

    for time_frame_index in time_frame:
        time_frame_stats['time_frame'].append(time_frame_index)
        time_frame_stats['slope'].append(time_frame[time_frame_index]['slope'])
        time_frame_stats['r2'].append(time_frame[time_frame_index]['r2'])
        time_frame_stats['percent_increase'].append(time_frame[time_frame_index]['percent_increase'])
        time_frame_stats['daily_percent_increase'].append(time_frame[time_frame_index]['daily_percent_increase'])

    time_source = ColumnDataSource(time_frame_stats)

    time_frame_stats_columns = [
        TableColumn(field="time_frame", title="Date"),
        TableColumn(field="slope", title="slope"),
        TableColumn(field="r2", title="r2"),
        TableColumn(field="percent_increase", title="percent_increase"),
        TableColumn(field="daily_percent_increase", title="daily_percent_increase"),
    ]

    time_frame_stats_data_table = DataTable(
        source=time_source,
        columns=time_frame_stats_columns,
        sizing_mode="stretch_width",
        height=280,
        index_position=None,
        margin=(-5, 80, -120, 80),
    )

    all_time_high_stats_title = Div(text="<h4>High Low Cycle Stats</h4>", margin=(5, 80, 5, 80))

    all_time_high_stats = {
        'date': [],
        'cycle_high': [],
        'cycle_low': [],
        'difference': [],
        'days': [],
        'percent': []
    }

    for ath in all_time_high:
        all_time_high_stats['date'].append(ath['date'].strftime("%Y-%m-%d"))
        all_time_high_stats['cycle_high'].append(ath['cycle_high'])
        all_time_high_stats['cycle_low'].append(ath['cycle_low'])
        all_time_high_stats['difference'].append(ath['difference'])
        all_time_high_stats['days'].append(ath['days'])
        all_time_high_stats['percent'].append(ath['percent'])

    all_time_source = ColumnDataSource(all_time_high_stats)

    all_time_high_stats_columns = [
        TableColumn(field="date", title="Date"),
        TableColumn(field="cycle_high", title="Cycle High"),
        TableColumn(field="cycle_low", title="Cycle Low"),
        TableColumn(field="difference", title="Difference"),
        TableColumn(field="days", title="Days"),
        TableColumn(field="percent", title="Percent"),
    ]

    all_time_high_stats_data_table = DataTable(
        source=all_time_source,
        columns=all_time_high_stats_columns,
        sizing_mode="stretch_width",
        height=280,
        index_position=None,
        margin=(-5, 80, -5, 80),
    )

    tabs.append(TabPanel(child=column(
        p_all,
        range_slider,
        select,
        p_downside,
        p_all_daily_percent_increase,
        p_all_rsi,
        p_macd,
        each_year_stats_title,
        each_year_stats_data_table,
        time_frame_stats_title,
        time_frame_stats_data_table,
        all_time_high_stats_title,
        all_time_high_stats_data_table,
        sizing_mode="stretch_width"
    ), title="all"))

    p_years = {}

    for year in list(each_year):
        p_years[year] = figure(
            x_axis_type="datetime",
            width=chart_width,
            height=chart_height,
            title="Multiple line example",
            x_axis_label="x",
            y_axis_label="y",
        )

        p_years[year].line(each_year[year]['DateTime'], each_year[year]['y'], legend_label="Value", color="blue", line_width=1)
        p_years[year].line(each_year[year]['DateTime'], each_year[year]['average_y_5'], legend_label="SMA 5", color="orange", line_width=1)
        p_years[year].line(each_year[year]['DateTime'], each_year[year]['average_y_20'], legend_label="SMA 20", color="green", line_width=1)
        p_years[year].line(each_year[year]['DateTime'], each_year[year]['average_y_60'], legend_label="SMA 60", color="red", line_width=1)

        p_years[year].xaxis.formatter = DatetimeTickFormatter(months="%b %Y")
        p_years[year].legend.click_policy = "hide"
        p_years[year].legend.location = "top_left"

        title = str(year) if year != list(each_year)[-1] else f"{year}-YTD"
        tabs.append(TabPanel(child=p_years[year], title=title))

    p_last_250 = figure(
        x_axis_type="datetime",
        width=chart_width,
        height=chart_height,
        title="Last 250 Days",
        x_axis_label="x",
        y_axis_label="y",
    )

    p_last_250.line(data_set['DateTime'][-250:], data_set['y'][-250:], legend_label="Value", color="blue", line_width=1)
    p_last_250.line(data_set['DateTime'][-250:], data_set['ma_5'][-250:], legend_label="SMA 5", color="orange", line_width=1)
    p_last_250.line(data_set['DateTime'][-250:], data_set['ma_20'][-250:], legend_label="SMA 20", color="green", line_width=1)
    p_last_250.line(data_set['DateTime'][-250:], data_set['ma_60'][-250:], legend_label="SMA 60", color="red", line_width=1)
    p_last_250.line(data_set['DateTime'][-250:], data_set['best_fit_line'][-250:], legend_label="linear", color="purple", line_width=1)
    p_last_250.line(data_set['DateTime'][-250:], data_set['theta_fit_list_2'][-250:], legend_label="poly 2", color="gold", line_width=1)
    p_last_250.line(data_set['DateTime'][-250:], data_set['best_fit_exp'][-250:], legend_label="Exponential", color="brown", line_width=1)

    p_last_250.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")
    p_last_250.legend.click_policy = "hide"
    p_last_250.legend.location = "top_left"

    tabs.append(TabPanel(child=p_last_250, title="last-250"))


    return tabs


def gen_bokeh_yoy_chart(year_over_year):
    
    tabs = []
    
    chart_width = 1500
    chart_height = 800

    colors = itertools.cycle(palette)

    p_year_over_years = figure(
        width=chart_width,
        height=chart_height,
        title="Multiple line example",
        x_axis_label="x",
        y_axis_label="y",
    )

    for year in list(year_over_year):
        label = str(year) if year != list(year_over_year)[-1] else f"{year}-ytd"
        p_year_over_years.line(year_over_year[year]['DateTime'], year_over_year[year]['y_values'], legend_label=label, color=next(colors), line_width=1)

    p_year_over_years.legend.click_policy = "hide"
    p_year_over_years.legend.location = "top_left"

    p_year_over_year_percents = figure(
        width=chart_width,
        height=chart_height,
        title="Year Over Year Percent",
        x_axis_label="x",
        y_axis_label="y",
    )

    colors = itertools.cycle(palette)  # Reset colors

    for year in list(year_over_year):
        label = str(year) if year != list(year_over_year)[-1] else f"{year}-ytd"
        p_year_over_year_percents.line(year_over_year[year]['DateTime'], year_over_year[year]['percent_increase'], legend_label=label, color=next(colors), line_width=1)

    p_year_over_year_percents.legend.click_policy = "hide"
    p_year_over_year_percents.legend.location = "top_left"

    tabs.append(TabPanel(child=column(p_year_over_years, p_year_over_year_percents), title="YoY"))

    return tabs


def gen_bokeh_benchmarks_chart(benchmarks):
    
    tabs = []
    
    p_benchmark = figure(
        sizing_mode="scale_width",
        aspect_ratio=11/5,
        title="Value",
        x_axis_label="x",
        y_axis_label="y",
    )

    p_benchmark.line(benchmarks['DateTime'], benchmarks['portfolio'], legend_label="Value", color="blue", line_width=1)
    p_benchmark.line(benchmarks['DateTime'], benchmarks['^GSPC'], legend_label="SP-500", color="orange", line_width=1)
    p_benchmark.line(benchmarks['DateTime'], benchmarks['^IXIC'], legend_label="NASDAQ", color="green", line_width=1)
    p_benchmark.line(benchmarks['DateTime'], benchmarks['^DJI'], legend_label="DJI", color="red", line_width=1)

    p_benchmark.legend.click_policy = "hide"
    p_benchmark.legend.location = "top_left"

    tabs.append(TabPanel(child=p_benchmark, title="Benchmark"))

    return tabs

all_start_time = time.perf_counter()

parser = argparse.ArgumentParser(description="Financial Data Analysis")

parser.add_argument("file_path", help="Path to CSV file")
parser.add_argument("--silent", action='store_true', help="Silence warnings")
parser.add_argument("--bokeh", action='store_true', help="Create Bokeh Charts")

args = parser.parse_args()

if args.silent:

    warnings.filterwarnings("ignore", category=optimize.OptimizeWarning)

    warnings.filterwarnings("ignore", category=ConvergenceWarning)

file_path = args.file_path

if not os.path.exists(file_path):
    print(f"Error: File '{file_path}' not found in {os.getcwd()}")
    print("Please provide a valid file path.")
    exit(1)

print()
print()
print("=====================================================")
print("processing ", file_path)
print("=====================================================")
print()

job_start_time = time.perf_counter()
all_time_df = gen_all_time(file_path)

#all_time['DateTime'] = all_time_df.index.tolist()

print('Generating base data                       {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))

job_start_time = time.perf_counter()
all_time_df = gen_all_time_downside(all_time_df)
print('Generating downside data                   {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))

job_start_time = time.perf_counter()
benchmarks = gen_benchmark(all_time_df)
print('Generating benchmark data                  {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))

job_start_time = time.perf_counter()
forecasts = gen_forecast(all_time_df)
print('Generating forecast data                   {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))

job_start_time = time.perf_counter()
all_time_high = gen_all_time_high(all_time_df)
print('Generating high low cycle data             {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))

job_start_time = time.perf_counter()
time_frame_stats = gen_time_frame_stats(all_time_df)
print('Generating time frame data                 {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))

job_start_time = time.perf_counter()
year_over_year = gen_year_over_year(all_time_df)


all_time = {k: all_time_df[k].tolist() for k in all_time_df.columns}

print('Generating year over year data             {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))

job_start_time = time.perf_counter()
all_time.update(gen_best_fit(all_time))
print('Generating best fit data                   {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))

job_start_time = time.perf_counter()
each_year = gen_each_year(all_time)
print('Generating each year data                  {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))

job_start_time = time.perf_counter()

for year in each_year:
    if len(each_year[year]['DateTime']) > 5:
        each_year[year].update(gen_best_fit(each_year[year]))

print('Generating each year best fit data         {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))

print(time.perf_counter() - all_start_time)

print()
print()

if args.bokeh:

    bokeh_time = time.perf_counter()

    job_start_time = time.perf_counter()
    tabs = gen_bokeh_chart(
        "all_time", all_time,
        each_year,
        time_frame_stats,
        benchmarks,
        all_time_high,
    )

    print('Generating main bokeh chart                 {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))

    job_start_time = time.perf_counter()
    tabs.extend(gen_bokeh_yoy_chart(year_over_year))
    print('Generating yoy bokeh chart                  {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))
        
        
    job_start_time = time.perf_counter()
    tabs.extend(gen_bokeh_benchmarks_chart(benchmarks))
    print('Generating benchmarks bokeh chart           {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))
        

    job_start_time = time.perf_counter()
    tabs.extend(gen_bokeh_forecast_chart(all_time, forecasts))
    print('Generating forecast bokeh chart             {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))
        
        
    job_start_time = time.perf_counter()

    tabs0 = Tabs(tabs=tabs)

    title_text = \
        "<h1>Date: " + all_time['DateTime'][-1].strftime("%Y-%m-%d") + "</h1>"
    title = Div(text=title_text, margin=(-10, 20, -10, 20))
    layout = column(children=[title, tabs0], sizing_mode="stretch_both")

    html_file = "financial_analysis.html"
    output_file(html_file, title="Financial Data Analysis", mode="inline")
    save(layout, resources=INLINE)

    full_path = os.path.abspath(html_file)
    webbrowser.open(f"file://{full_path}")

    print('Render bokeh chart                          {0:>3.5f}'.format(
        time.perf_counter() - job_start_time,
        ))

    print()

    print('Total runtime                               {0:>3.5f}'.format(
        time.perf_counter() - all_start_time,
        ))