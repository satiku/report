import csv
import argparse
import numpy as np
import os
import ta
import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import Tabs, Tooltip, Panel
from bokeh.models import DatetimeTickFormatter
from bokeh.models import tickers
from bokeh.layouts import column
from bokeh.models import RangeTool
from bokeh.models import BoxAnnotation
from bokeh.models import LinearAxis, Range1d
from bokeh.models import RangeSlider
from bokeh.models import ColumnDataSource, DataTable, TableColumn, Div
from bokeh.palettes import Dark2_5 as palette
from bokeh.models import HoverTool
from bokeh.models import CrosshairTool, Span
import itertools
import time
from datetime import datetime 
from datetime import timedelta 
from matplotlib.dates import date2num
from scipy.optimize import curve_fit
import yfinance as yf
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def moving_average(window, inputValue):
    
    average_y = []

    for ind in range(len(inputValue) - window + 1):
        average_y.append(np.mean(inputValue[ind:ind+window]))


    for ind in range(window - 1):
        average_y.insert(0, np.nan)


    return average_y


def exponential_func(x, a, b, c):
    return a * np.exp(b * x) + c

def log_exponential_func(x, lna, b, c):
    return  lna + b * x  


def gen_all_time(file):
    
    all_time = {}

    all_time['x_values'] = []
    all_time['y_values'] = []  
    all_time['daily_percent_increase'] = []

    all_time['average_y_5_20'] = []
    all_time['average_y_5_60'] = []                
    all_time['average_y_20_60'] = []


    with open(file, newline='') as csvfile:

    #with open('Net Worth.csv', newline='') as csvfile:


        previous_value = float(0)

        reader = csv.reader(csvfile)

        for row in reader:
            if row[1] != "" and row[1] != "Date":

                if previous_value == 0 : 
                    previous_value = float(row[2])

                all_time['x_values'].append( datetime.strptime(row[1].split(" ")[0], '%Y-%m-%d'))
                all_time['y_values'].append(float(row[2]))
               

                all_time['daily_percent_increase'].append(((float(row[2]) - previous_value)/previous_value)*100)
                
                previous_value = float(row[2])

        all_time['average_y_5']  = moving_average(5 ,all_time['y_values'])
        all_time['average_y_20'] = moving_average(20 ,all_time['y_values'])
        all_time['average_y_60'] = moving_average(60 ,all_time['y_values'])
        
        all_time['daily_percent_increase_ma_5']   = moving_average(  5 ,all_time['daily_percent_increase'])
        all_time['daily_percent_increase_ma_20']  = moving_average( 20 ,all_time['daily_percent_increase'])
        all_time['daily_percent_increase_ma_60']  = moving_average( 60 ,all_time['daily_percent_increase'])
        all_time['daily_percent_increase_ma_250'] = moving_average(250 ,all_time['daily_percent_increase'])
        
                
        
        

        for index ,_ in enumerate(all_time['y_values']) :
            
            if np.isnan(all_time['average_y_5'][index]) : 
                all_time['average_y_5_20'].append(np.nan)
                all_time['average_y_5_60'].append(np.nan)               
                all_time['average_y_20_60'].append(np.nan) 
                
                continue
                
            elif np.isnan(all_time['average_y_20'][index] ):
                all_time['average_y_5_20'].append(np.nan)
                all_time['average_y_5_60'].append(np.nan)               
                all_time['average_y_20_60'].append(np.nan)                
                
                continue 
                
                
            elif np.isnan(all_time['average_y_60'][index]):
                all_time['average_y_5_20'].append(all_time['average_y_5'][index] - all_time['average_y_20'][index])
                
                
                all_time['average_y_5_60'].append(np.nan)    
                all_time['average_y_20_60'].append(np.nan) 
                
                continue 
                
                
            else: 
                all_time['average_y_5_20'].append(all_time['average_y_5'][index] - all_time['average_y_20'][index])
                all_time['average_y_5_60'].append(all_time['average_y_5'][index] - all_time['average_y_60'][index])
                all_time['average_y_20_60'].append(all_time['average_y_20'][index] - all_time['average_y_60'][index])    

        df = pd.DataFrame({'Actual': all_time['y_values']})

        all_time['rsi'] = list(ta.momentum.RSIIndicator(df['Actual']).rsi())
        
        
        
        
    return all_time



def gen_each_year(all_time):

    each_year = {}

    for index, value in enumerate(all_time['y_values']) :

        year =all_time['x_values'][index].year
        
        if year not in each_year:
            each_year[year] = {}

            each_year[year]['x_values'] = []
            each_year[year]['y_values'] = []
            
            each_year[year]['average_y_5'] = []
            each_year[year]['average_y_20'] = []
            each_year[year]['average_y_60'] = []
            
            each_year[year]['average_y_5_20'] = []
            each_year[year]['average_y_5_60'] = []                
            each_year[year]['average_y_20_60'] = []                
            
            each_year[year]['rsi'] = []   
            
            each_year[year]['daily_percent_increase'] = []
            
            each_year[year]['daily_percent_increase_ma_5']   = []
            each_year[year]['daily_percent_increase_ma_20']  = []
            each_year[year]['daily_percent_increase_ma_60']  = []
            each_year[year]['daily_percent_increase_ma_250'] = []


        each_year[year]['x_values'].append(all_time['x_values'][index])
        each_year[year]['y_values'].append(value)
        
        each_year[year]['average_y_5'].append(all_time['average_y_5'][index])
        each_year[year]['average_y_20'].append(all_time['average_y_20'][index])
        each_year[year]['average_y_60'].append(all_time['average_y_60'][index])
        
        each_year[year]['average_y_5_20'].append(all_time['average_y_5'][index] - all_time['average_y_20'][index])
        each_year[year]['average_y_5_60'].append(all_time['average_y_5'][index] - all_time['average_y_60'][index])
        each_year[year]['average_y_20_60'].append(all_time['average_y_20'][index] - all_time['average_y_60'][index])    
        
        each_year[year]['rsi'].append(all_time['rsi'][index])
        
        each_year[year]['daily_percent_increase'].append(all_time['daily_percent_increase'][index])
            
        each_year[year]['daily_percent_increase_ma_5'].append(all_time['daily_percent_increase_ma_5'][index])
        each_year[year]['daily_percent_increase_ma_20'].append(all_time['daily_percent_increase_ma_20'][index])
        each_year[year]['daily_percent_increase_ma_60'].append(all_time['daily_percent_increase_ma_60'][index])
        each_year[year]['daily_percent_increase_ma_250'].append(all_time['daily_percent_increase_ma_250'][index])



    return each_year



def gen_year_over_year(all_time):

    year_over_year = {}

    for index, value in enumerate(all_time['y_values']) :

        year =all_time['x_values'][index].year
        
        if year not in year_over_year:
            year_over_year[year] = {}

            year_over_year[year]['x_values'] = []
            year_over_year[year]['y_values'] = []
            year_over_year[year]['percent_increase'] = []
            
            starting_value = value
            starting_index = index

        year_over_year[year]['x_values'].append(index - starting_index)
        year_over_year[year]['y_values'].append(value - starting_value)
        year_over_year[year]['percent_increase'].append(round((((value - starting_value)/ starting_value )*100),3 ))
        

    return year_over_year




def gen_all_time_high(all_time):

    all_time_high = []
    
    cycle_high = 0
    cycle_low = all_time['y_values'][0]

 
    for index, value in enumerate(all_time['y_values']) :

        if value > cycle_high : 
            
            if cycle_high != cycle_low and index != 0 and cycle_low_index - cycle_high_index >= 2:
                item = {}
                
                
                item['date'] = all_time['x_values'][cycle_high_index] 
                item['cycle_high'] = round(cycle_high,3)
                item['cycle_low'] = cycle_low 
                item['difference'] = round(cycle_high - cycle_low ,3) 
                item['days'] = cycle_low_index - cycle_high_index 
                item['percent'] = round((((cycle_low - cycle_high)/cycle_high)*100),3)
                
                all_time_high.append(item)
                
                
                
            cycle_high_index = index
            cycle_high = value
            cycle_low = value
            
        elif value < cycle_high and value < cycle_low : 
            
            cycle_low = value
            cycle_low_index = index 
    
    
    return all_time_high




def gen_all_time_downside(all_time):

    all_time_downside = {}
    
    
    
    all_time_downside['downside'] = []
    all_time_downside['downside_percent'] = []
            
            
            
            
    cycle_high = 0

 
    for index, value in enumerate(all_time['y_values']) :

        if value > cycle_high : 
            
            cycle_high = value
            
            
            
            all_time_downside['downside'].append(0) 
            all_time_downside['downside_percent'].append(0) 
            
            
        elif value < cycle_high : 
            
            
            
            
            all_time_downside['downside'].append(round(cycle_high - value ,3) * -1 ) 
            all_time_downside['downside_percent'].append(round((((value - cycle_high)/cycle_high)*100),3) ) 
  
    
    return all_time_downside




def gen_best_fit(data_set):

    best_fit = {}



    converted_dates = date2num(data_set['x_values']) 


    x = converted_dates
    y = data_set['y_values']
    
    
    
    sst = sum([(day - np.average(y))**2 for day in y ])

    m_b,ssr , _, _, _ = np.polyfit(np.array(range(len(x))), y, 1,full="true")

    m = m_b[0]
    b = m_b[1]
    ssr =ssr[0]
    
    r2 = 1 -(ssr/sst)
    
    theta_2 = np.polyfit(np.array(range(len(x))), y, 2)
    
    
    
    percent_increase = ((y[-1] - y[0])/y[0])*100
    percent_increase_daily_avg = percent_increase/len(y)


    
    best_fit_line = []
    theta_fit_list_2 =[]

    
    for i in range(len(data_set['x_values'])):

        best_fit_line.append(m*i+b)
        
        
        z_2 = np.poly1d(theta_2)
        
        theta_fit_list_2.append(z_2(i))
    

    
    best_fit['slope'] = round(m, 3)
    best_fit['r2'] = round(r2, 3)
    best_fit['percent_increase'] = round(percent_increase, 3)
    best_fit['percent_increase_daily_avg'] = round(percent_increase_daily_avg, 3)
    
    best_fit['best_fit_line'] = best_fit_line
    best_fit['theta_fit_list_2'] = theta_fit_list_2   
    
    
    
    
    # Initial guess for the parameters
    p0 = (1, 1, 1)  # a, b, c


    y_transformed = np.log(np.maximum(np.array(y) - 1, 1e-10))
    # Fit the data using curve_fit
    params, cov = curve_fit(log_exponential_func, x, y_transformed, p0)
    lna, b, c = params
    a = np.exp(lna)
            
    x_fit = np.linspace(min(x), max(x), len(x))
    
    y_fit = exponential_func(x_fit, a, b, c)


    
    best_fit['best_fit_exp'] = y_fit
    return(best_fit)






def gen_benchmark(data_set):

    benchmark = {}
    tickers = ["^GSPC","^IXIC","^DJI"]
    
    benchmark['portfolio'] = []
    benchmark['x_values'] = []    
    
    benchmark_data_set = {}
    
    print(len(data_set['x_values']))

    benchmark_data_set['x_values'] = data_set['x_values'][-60:]
    benchmark_data_set['y_values'] = data_set['y_values'][-60:]

    print(len(benchmark_data_set['x_values']))




    data = yf.download(tickers, start = benchmark_data_set['x_values'][0] , auto_adjust=True, progress=False)
    
    new_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
    data = data.reindex(new_index)
    
    data.interpolate(method='time', inplace=True)
    
    
    
    starting_value = benchmark_data_set['y_values'][0]
    
    for index, date in enumerate(benchmark_data_set['x_values']):

        value = benchmark_data_set['y_values'][index]
        
        benchmark['portfolio'].append(round((((value - starting_value)/ starting_value )*100),3 ))
        benchmark['x_values'].append(index)

    
    
    for ticker in tickers :
        benchmark[ticker] = []
        
        print(ticker)
        starting_value = data.loc[benchmark_data_set['x_values'][0], ('Close', ticker)]
        
        for date in benchmark_data_set['x_values']:

            value = data.loc[date, ('Close', ticker)]
            benchmark[ticker].append(round((((value - starting_value)/ starting_value )*100),3 ))








    return(benchmark)






def gen_forecast(data_set): 
    
    forecast_time = 250
    
    forcasted_values = {}
    forcasted_values['x_values'] = []
    forcasted_values['y_values_add'] = []
    forcasted_values['y_values_mul'] = []
    forcasted_values['y_values_avg'] = []
    
    date = data_set['x_values'][-1]
    
    model_add = ExponentialSmoothing(data_set['y_values'], seasonal_periods=250, trend='add', seasonal='add' )
    model_mul = ExponentialSmoothing(data_set['y_values'], seasonal_periods=250, trend='mul', seasonal='mul' )
    
    fitted_model_add = model_add.fit()    
    fitted_model_mul = model_mul.fit()    
    
    # Forecast for the next 12 months
    forecast_add = fitted_model_add.forecast(steps=forecast_time)
    forecast_mul = fitted_model_mul.forecast(steps=forecast_time)
    
    forcasted_values['y_values_add'].extend(list(forecast_add))
    forcasted_values['y_values_mul'].extend(list(forecast_mul))
        
    
    for index in range(forecast_time) :
        forcasted_values['y_values_avg'].append((forcasted_values['y_values_add'][index] + forcasted_values['y_values_mul'][index])/2)
        
        
        if date.weekday() == 4 :
            date = date + timedelta(days = 3)
            forcasted_values['x_values'].append(date)
        else : 
            date = date + timedelta(days = 1)
            forcasted_values['x_values'].append(date)
        
        
    return forcasted_values




def gen_time_frame_stats(all_time):
    
    
    time_frame_stats = {} 

    time_frame_stats['5'] = {}   
    time_frame_stats['20'] = {}
    time_frame_stats['60'] = {}
    time_frame_stats['250'] = {}



    
    y_values = all_time['y_values']
    
    
    
    for time_frame in time_frame_stats :
    
        time_frame_stats[time_frame]['percent_increase']       = round((((y_values[-1] - y_values[-1*int(time_frame)])/y_values[-1*int(time_frame)])*100), 3)
        time_frame_stats[time_frame]['daily_percent_increase'] = round((((y_values[-1] - y_values[-1*int(time_frame)])/y_values[-1*int(time_frame)])*100)/int(time_frame), 3)

        m_b,ssr , _, _, _ = np.polyfit(np.array(range(len(y_values[(-1* int(time_frame)):]))), y_values[(-1* int(time_frame)):], 1,full="true")

        sst = sum([(day - np.average(y_values[(-1* int(time_frame)):]))**2 for day in y_values[(-1* int(time_frame)):] ])

        ssr =ssr[0]
        
        
        m = m_b[0]        
        r2 = 1 -(ssr/sst)
        
        time_frame_stats[time_frame]['slope'] = round(m, 3)
        time_frame_stats[time_frame]['r2'] = round(r2, 3)
        
    
        
    return(time_frame_stats)
    


def gen_bokeh_chart(data_set_id, data_set, each_year, time_frame, year_over_year, benchmarks, forecasts):
    
    chart_width = 1500
    chart_height = 800
    
    
    tabs = []
    


    date_buffer = int((data_set['x_values'][-1] - data_set['x_values'][0]).days * 0.025)
    
    
    source = ColumnDataSource(data={
    'x': data_set['x_values'],
    'y_value': data_set['y_values'],
    'y_sma5': data_set['average_y_5'],
    'y_sma20': data_set['average_y_20'],
    'y_sma60': data_set['average_y_60']
    # Add more fields for other lines if needed
})
    
    

    
    
    p_all = figure(
        x_axis_type="datetime", 
        sizing_mode="scale_width", 
        aspect_ratio=11/5 ,
        title="Value", 
        x_axis_label="x", 
        y_axis_label="y", 
        x_range=(data_set['x_values'][0] - timedelta(days = date_buffer), data_set['x_values'][-1] + timedelta(days = date_buffer))
        )


    hover = HoverTool(
        tooltips=[
            ("Date", "@x{%F}"),
            ("Tool", "$y{0,0.00}"),
            ("Value", "@y_value{0,0.00}"),
            ("SMA 5", "@y_sma5{0,0.00}"),
            ("SMA 20", "@y_sma20{0,0.00}"),
            ("SMA 60", "@y_sma60{0,0.00}"),            
        ],
        formatters={'@x': 'datetime'},
        mode='vline'        
    )
    cross = CrosshairTool()
    p_all.add_tools(hover,cross)

    
    
    
    
    
    
    
            
    # add multiple renderers
    p_all.line('x', 'y_value', source=source, legend_label="Value", color="blue", line_width=1, name="Value")
    p_all.line('x', 'y_sma5', source=source, legend_label="SMA 5", color="orange", line_width=1, name="SMA 5")
    p_all.line('x', 'y_sma20', source=source, legend_label="SMA 20", color="green", line_width=1, name="SMA 20")
    p_all.line('x', 'y_sma60', source=source, legend_label="SMA 60", color="red", line_width=1, name="SMA 60")
        

  


  
  
  
  
  
  
  
  
  
  
  
  
    p_all.xaxis[0].formatter = DatetimeTickFormatter(days=["%m - %Y"], months=["%m - %Y"],)  
    
    p_all.xaxis.ticker = tickers.MonthsTicker(months=list(range(0, 13, 1)))
    p_all.xaxis.major_label_orientation = 0.9
    
    p_all.legend.click_policy="hide"
    p_all.legend.location = "top_left"    
    
    
    
    
    range_slider = RangeSlider(
        title="Adjust y-axis range",
        start=0,
        end=max(data_set['y_values'])+ (max(data_set['y_values'])*.1),
        step=1,
        value=(0,data_set['y_values'][-1]),
        margin=(5,80,5,80)
    )
    range_slider.js_link("value", p_all.y_range, "start", attr_selector=0)
    range_slider.js_link("value", p_all.y_range, "end", attr_selector=1)
        
        
        
        
        
        
    
   
    p_all_daily_percent_increase = figure(
        x_axis_type="datetime", 
        sizing_mode="scale_width", 
        aspect_ratio= 10,
        title="Daily Percent Increase", 
        y_axis_label="y" ,
        x_range=p_all.x_range , 
        y_range=(-5,5))

    # add multiple renderers
    p_all_daily_percent_increase.line(data_set['x_values'], data_set['daily_percent_increase'], legend_label="Value", color="blue", line_width=1)
    
    p_all_daily_percent_increase.line(data_set['x_values'], data_set['daily_percent_increase_ma_5'] , legend_label="SMA 5", color="orange", line_width=1)
    p_all_daily_percent_increase.line(data_set['x_values'], data_set['daily_percent_increase_ma_20'], legend_label="SMA 20", color="green", line_width=1)
    p_all_daily_percent_increase.line(data_set['x_values'], data_set['daily_percent_increase_ma_60'], legend_label="SMA 60", color="red", line_width=1)
    p_all_daily_percent_increase.line(data_set['x_values'], data_set['daily_percent_increase_ma_250'], legend_label="SMA 250", color="purple", line_width=1)

    p_all_daily_percent_increase.xaxis.ticker = tickers.MonthsTicker(months=list(range(0, 13, 1)))    
    p_all_daily_percent_increase.xaxis.major_label_text_font_size = '0pt'
    
    p_all_daily_percent_increase.legend.click_policy="hide"
    p_all_daily_percent_increase.legend.location = "top_left"    
      
    
    
    
    p_all_rsi = figure(
        x_axis_type="datetime", 
        sizing_mode="scale_width", 
        aspect_ratio=10, 
        title="RSI", 
        y_axis_label="y" ,
        x_range=p_all.x_range,
        y_range=(0,100),
        )

    # add multiple renderers
    p_all_rsi.line(data_set['x_values'], data_set['rsi'], legend_label="RSI", color="blue", line_width=1)
    

    p_all_rsi.add_layout(BoxAnnotation(top=40, fill_alpha=0.1, fill_color='green', line_color='green'))
    p_all_rsi.add_layout(BoxAnnotation(bottom=80, fill_alpha=0.1, fill_color='red', line_color='red'))


    p_all_rsi.xaxis.ticker = tickers.MonthsTicker(months=list(range(0, 13, 1)))
    p_all_rsi.xaxis.major_label_text_font_size = '0pt'
    
    p_all_rsi.legend.click_policy="hide"
    p_all_rsi.legend.location = "top_left"    
          
    






    
    
    
    p_macd = figure(x_axis_type="datetime", sizing_mode="scale_width", aspect_ratio=7, title="MACD", x_axis_label="x", y_axis_label="y" ,x_range=p_all.x_range)

    # add multiple renderers

    p_macd.line(data_set['x_values'], data_set['average_y_5_20']     , legend_label="Running average 5 - 20"  ,   color="green" , line_width=1 )
#    plt.plot_date(data_set['x_values'], data_set['average_y_5_60']  , legend_label='Running average 5 - 60'   , color='orange' , line_width=1 )
    p_macd.line(data_set['x_values'], data_set['average_y_20_60']    , legend_label="Running average 20 - 60"   , color="red"   , line_width=1     )


        
    p_macd.xaxis[0].formatter = DatetimeTickFormatter(days=["%m - %Y"], months=["%m - %Y"],)
    
    p_macd.xaxis.ticker = tickers.MonthsTicker(months=list(range(0, 13, 1)))
    p_macd.xaxis.major_label_orientation = 0.9
    
    p_macd.legend.click_policy="hide"
    p_macd.legend.location = "top_left"    
          
    




  
    
    p_downside = figure(
        x_axis_type="datetime", 
        sizing_mode="scale_width", 
        aspect_ratio=10 ,
        title="Downside", 
        y_axis_label="y" ,
        x_range=p_all.x_range , 
        y_range=(-2500, 100)
        )

    

    # Setting the second y axis range name and range
    p_downside.extra_y_ranges = {"foo": Range1d(start=-25, end=1)}

    # Adding the second axis to the plot.  
    p_downside.add_layout(LinearAxis(y_range_name="foo"), 'right')




    # add multiple renderers
    p_downside.line(data_set['x_values'], data_set['downside'], legend_label="Downside", color="blue", line_width=1)
    p_downside.line(data_set['x_values'], data_set['downside_percent'],
        legend_label="Downside Percent", 
        color="green", 
        line_width=1 , 
        y_range_name="foo"
        )    

        
    p_downside.xaxis[0].formatter = DatetimeTickFormatter(days=["%m - %Y"], months=["%m - %Y"],)
    
    p_downside.xaxis.ticker = tickers.MonthsTicker(months=list(range(0, 13, 1)))
    p_downside.xaxis.major_label_text_font_size = '0pt'
    
    
    p_downside.legend.click_policy="hide"
    p_downside.legend.location = "bottom_left"    
    











    select = figure(title="Drag the middle and edges of the selection box to change the range above",
                    height=130, width=800, 
                    x_axis_type="datetime", y_axis_type=None,
                    tools="", toolbar_location=None, background_fill_color="#efefef")

    range_tool = RangeTool(x_range=p_all.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.2

    select.line(data_set['x_values'], data_set['y_values'])
    select.ygrid.grid_line_color = None
    select.add_tools(range_tool)





# Generate Data Tables 
#
#
#
#
    each_year_stats_title = Div(text="<h4>Yearly Stats</h4>", margin=(5,80,5,80),)


    each_year_stats = {}
    
    each_year_stats['year'] = []
    each_year_stats['slope'] = []
    each_year_stats['r2'] = []
    each_year_stats['percent_increase'] = []
    each_year_stats['percent_increase_daily_avg'] = []
    each_year_stats['total_days'] = []

    
    for year in each_year:
        
        
        each_year_stats['year'].append(year) 
        each_year_stats['slope'].append(each_year[year]['slope'])
        each_year_stats['r2'].append(each_year[year]['r2'])
        each_year_stats['percent_increase'].append(each_year[year]['percent_increase'])
        each_year_stats['percent_increase_daily_avg'].append(each_year[year]['percent_increase_daily_avg'])
        each_year_stats['total_days'].append(len(each_year[year]['y_values']))
        
      
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
        width=400, 
        height=280 , 
        index_position = None , 
        margin=(-5,80,-120,80),
        )





    time_frame_stats_title = Div(text="<h4>Time Frame Stats</h4>", margin=(5,80,5,80),)

    time_frame_stats = {}


    time_frame_stats['time_frame'] = []
    time_frame_stats['slope'] = []
    time_frame_stats['r2'] = []
    time_frame_stats['percent_increase'] = []
    time_frame_stats['daily_percent_increase'] = []


    

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
        width=400, 
        height=280 , 
        index_position = None , 
        margin=(-5,80,-5,80),
        )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    tabs.append(Panel(child=column(
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
        sizing_mode="stretch_width"
        ), title="all"))
    
    
    
    
    
    
    
    
    p_years = {}
    
    for year in list(each_year):
    
        print(year)

        
        
        p_years['year'] = figure(x_axis_type="datetime", width=chart_width, height=chart_height,title="Multiple line example", x_axis_label="x", y_axis_label="y")
        
        
        
        # add multiple renderers
        p_years['year'].line(each_year[year]['x_values'], each_year[year]['y_values'], legend_label="Value", color="blue", line_width=1)
        
        p_years['year'].line(each_year[year]['x_values'], each_year[year]['average_y_5'] , legend_label="SMA 5", color="orange", line_width=1)
        p_years['year'].line(each_year[year]['x_values'], each_year[year]['average_y_20'], legend_label="SMA 20", color="green", line_width=1)
        p_years['year'].line(each_year[year]['x_values'], each_year[year]['average_y_60'], legend_label="SMA 60", color="red", line_width=1)
        # show the results
        
        p_years['year'].xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")
        
        p_years['year'].legend.click_policy="hide"
        p_years['year'].legend.location = "top_left"    
    
    
        if year !=  list(each_year)[-1:] :
            tabs.append(Panel(child=p_years['year'], title=str(year)))
        else :
            tabs.append(Panel(child=p_years['year'], title=str(year)+"-YTD"))




  
    
    p_all = figure(x_axis_type="datetime", width=chart_width, height=chart_height,title="Multiple line example", x_axis_label="x", y_axis_label="y")

    # add multiple renderers
    p_all.line(data_set['x_values'][-250:], data_set['y_values'][-250:] , legend_label="Value", color="blue", line_width=1)
    
    p_all.line(data_set['x_values'][-250:], data_set['average_y_5'][-250:]  , legend_label="SMA 5", color="orange", line_width=1)
    p_all.line(data_set['x_values'][-250:], data_set['average_y_20'][-250:] , legend_label="SMA 20", color="green", line_width=1)
    p_all.line(data_set['x_values'][-250:], data_set['average_y_60'][-250:] , legend_label="SMA 60", color="red", line_width=1)
    # show the results
    
    p_all.line(data_set['x_values'][-250:], data_set['best_fit_line'][-250:] , legend_label="linear", color="purple", line_width=1)
    p_all.line(data_set['x_values'][-250:], data_set['theta_fit_list_2'][-250:] , legend_label="poly 2", color="gold", line_width=1)    
  
        
    p_all.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")
    
    p_all.legend.click_policy="hide"
    p_all.legend.location = "top_left"    
    
    
    tabs.append(Panel(child=p_all, title="last-250"))








    
    #colors has a list of colors which can be used in plots 
    colors = itertools.cycle(palette) 

    
    p_year_over_years = figure(width=chart_width, height=chart_height,title="Multiple line example", x_axis_label="x", y_axis_label="y")
    
    
    for index, year in enumerate(list(year_over_year)):


    
        

    
        if year !=  list(year_over_year)[-1:] :
            p_year_over_years.line(year_over_year[year]['x_values'], year_over_year[year]['y_values'], legend_label=str(year), color=next(colors), line_width=1)
        
        else :
            p_year_over_years.line(year_over_year[year]['x_values'], year_over_year[year]['y_values'] , legend_label=(str(year) + "-ytd"), color=next(colors), line_width=1)
        
        
        p_year_over_years.legend.click_policy="hide"
        p_year_over_years.legend.location = "top_left"    




    p_year_over_year_percents = figure(width=chart_width, height=chart_height,title="Year Over Year Percent", x_axis_label="x", y_axis_label="y")
    
    
    for index, year in enumerate(list(year_over_year)):
    

    

    
    
        if year !=  list(year_over_year)[-1:] :
            p_year_over_year_percents.line(year_over_year[year]['x_values'], year_over_year[year]['percent_increase'], legend_label=str(year), color=next(colors), line_width=1)
        
        else :
            p_year_over_year_percents.line(year_over_year[year]['x_values'], year_over_year[year]['percent_increase'] , legend_label=(str(year) + "-ytd"), color=next(colors), line_width=1)
        
        
        
        p_year_over_year_percents.legend.click_policy="hide"
        p_year_over_year_percents.legend.location = "top_left"    



    tabs.append(Panel(child=column(p_year_over_years, p_year_over_year_percents), title="YoY"))










    p_benchmark = figure(
        sizing_mode="scale_width", 
        aspect_ratio=11/5 ,
        title="Value", 
        x_axis_label="x", 
        y_axis_label="y", 
        )

    # add multiple renderers
    p_benchmark.line(benchmarks['x_values'], benchmarks['portfolio'], legend_label="Value", color="blue", line_width=1)
    
    p_benchmark.line(benchmarks['x_values'], benchmarks['^GSPC'] , legend_label="SP-500", color="orange", line_width=1)
    p_benchmark.line(benchmarks['x_values'], benchmarks['^IXIC'], legend_label="NASDAQ", color="green", line_width=1)
    p_benchmark.line(benchmarks['x_values'], benchmarks['^DJI'], legend_label="DJI", color="red", line_width=1)
    # show the results
    

    
    p_benchmark.legend.click_policy="hide"
    p_benchmark.legend.location = "top_left"    
    

    tabs.append(Panel(child=p_benchmark, title="Benchmark"))












    p_forecast = figure(
        x_axis_type="datetime", 
        sizing_mode="scale_width", 
        aspect_ratio=11/5 ,
        title="Value", 
        x_axis_label="x", 
        y_axis_label="y", 
        )

    # add multiple renderers
    p_forecast.line(data_set['x_values'], data_set['y_values'], legend_label="Value", color="blue", line_width=1)
    
    p_forecast.line(forecasts['x_values'], forecasts['y_values_add'] , legend_label="Holt-Winters add", color="orange", line_width=1)
    p_forecast.line(forecasts['x_values'], forecasts['y_values_mul'] , legend_label="Holt-Winters mul", color="red", line_width=1)    
    p_forecast.line(forecasts['x_values'], forecasts['y_values_avg'] , legend_label="Holt-Winters avg", color="green", line_width=1)        

    p_forecast.xaxis[0].formatter = DatetimeTickFormatter(days=["%m - %Y"], months=["%m - %Y"],)
    p_forecast.xaxis.ticker = tickers.MonthsTicker(months=list(range(0, 13, 1)))
    p_forecast.xaxis.major_label_orientation = 0.9
    
    p_forecast.legend.click_policy="hide"
    p_forecast.legend.location = "top_left"    
    

    tabs.append(Panel(child=p_forecast, title="Forecast"))

















    tabs0 = Tabs(tabs=tabs)
    
    
    
    show(tabs0)



    
    
    
    
    
# Initialize the ArgumentParser
parser = argparse.ArgumentParser(description="A program to demonstrate optional arguments.")

# Define optional arguments
#parser.add_argument("-s", "--short", help="This is a short option")
parser.add_argument("--ytd",    action='store_true', help="Run current YTD calculations")
parser.add_argument("--all",    action='store_true', help="Run all time calculations")
parser.add_argument("--years",  action='store_true', help="Run all past years calculations")
parser.add_argument("--bokeh",  action='store_true', help="Create Bokeh Charts")

# Parse the arguments
args = parser.parse_args()

    
    
    
    

all_start_time = time.perf_counter()

all_time = gen_all_time('P1.csv')
all_time.update(gen_best_fit(all_time))
    
all_time.update(gen_all_time_downside(all_time))

benchmarks = gen_benchmark(all_time)

forecasts = gen_forecast(all_time)

print(time.perf_counter() - all_start_time)


each_year = gen_each_year(all_time)


all_time_high = gen_all_time_high(all_time)


time_frame_stats = gen_time_frame_stats(all_time)


year_over_year = gen_year_over_year(all_time)


for year in each_year:
    
    
    each_year[year].update(gen_best_fit(each_year[year]))
    

print()
print()
print("=============================")
print(" high low cycle")
print("=============================")
print()
print("------------+------------------+------------------+--------------+--------+----------+")
print("        date|     all time high|      all time low|    difference|    days|   percent|")
print("------------+------------------+------------------+--------------+--------+----------+")


for item in all_time_high :


    print('{0:>12s}|  {1:>16.2f}|  {2:>16.2f}|  {3:>12.2f}|  {4:6}|  {5:>8.2f}|'.format(
    
        str(item['date'].strftime("%Y-%m-%d")) , 
        item['cycle_high'] , 
        item['cycle_low'] , 
        item['difference'] , 
        item['days'] , 
        item['percent']
    
    ))




if args.bokeh:
    
    print()
    print("=============================")
    print("Generating Bokeh")
    print("=============================")


    bokeh_time = time.perf_counter()
    
    gen_bokeh_chart("all_time", all_time , each_year, time_frame_stats, year_over_year, benchmarks, forecasts)


    print(time.perf_counter() - bokeh_time)




