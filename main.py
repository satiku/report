import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import ta
import pandas as pd
from  fpdf import FPDF as fpdf
import time
from datetime import datetime 
from matplotlib.dates import ( 
    DateFormatter, AutoDateLocator, AutoDateFormatter, datestr2num 
) 


def moving_average(window, inputValue):
    
    average_y = []

    for ind in range(len(inputValue) - window + 1):
        average_y.append(np.mean(inputValue[ind:ind+window]))


    for ind in range(window - 1):
        average_y.insert(0, np.nan)


    return average_y



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

                all_time['x_values'].append(row[1].split(" ")[0])
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

            all_time['average_y_5_20'].append(all_time['average_y_5'][index] - all_time['average_y_20'][index])
            all_time['average_y_5_60'].append(all_time['average_y_5'][index] - all_time['average_y_60'][index])
            all_time['average_y_20_60'].append(all_time['average_y_20'][index] - all_time['average_y_60'][index])    

        df = pd.DataFrame({'Actual': all_time['y_values']})

        all_time['rsi'] = list(ta.momentum.RSIIndicator(df['Actual']).rsi())
        
        
        
        
    return all_time



def gen_each_year(all_time):

    each_year = {}

    for index, value in enumerate(all_time['y_values']) :

        year =all_time['x_values'][index].split(" ")[0].split("-")[0]
        
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



def gen_all_time_high(all_time):

    all_time_high = []
    
    cycle_high = 0
    cycle_low = all_time['y_values'][0]

 
    for index, value in enumerate(all_time['y_values']) :

        if value > cycle_high : 
            
            if cycle_high != cycle_low and index != 0 and cycle_low_index - cycle_high_index > 2:
                item = {}
                
                
                item['date'] = all_time['x_values'][cycle_high_index] 
                item['cycle_high'] = round(cycle_high,3)
                item['cycle_low'] = cycle_low 
                item['difference'] = round(cycle_high - cycle_low ,3) 
                item['days'] = cycle_low_index - cycle_high_index 
                item['percent'] = round((((cycle_high - cycle_low)/cycle_high)*100),3)
                
                all_time_high.append(item)
                
                
                
            cycle_high_index = index
            cycle_high = value
            cycle_low = value
            
        elif value < cycle_high and value < cycle_low : 
            
            cycle_low = value
            cycle_low_index = index 
    
    
    return all_time_high



def gen_best_fit(data_set):

    best_fit = {}



    converted_dates = datestr2num([ datetime.strptime(day, '%Y-%m-%d').strftime('%m/%d/%Y') for day in data_set['x_values'] ]) 


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
    


    
    best_fit_line = []
    theta_fit_list_2 =[]

    
    for i in range(len(data_set['x_values'])):

        best_fit_line.append(m*i+b)
        
        
        z_2 = np.poly1d(theta_2)
        
        theta_fit_list_2.append(z_2(i))
    

    
    best_fit['slope'] = round(m, 3)
    best_fit['r2'] = round(r2, 3)
    best_fit['percent_increase'] = round(percent_increase, 3)
    
    best_fit['best_fit_line'] = best_fit_line
    best_fit['theta_fit_list_2'] = theta_fit_list_2   
    
    
    return(best_fit)



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
    


def gen_best_fit_ma_chart(data_set_id, data_set):
    

#    
#    x = np.array(converted_dates)
#    y = np.array(each_year[year]['y_values'])
#
#    
#    average_y_5 = np.array(each_year[year]['average_y_5'])
#    average_y_20 = np.array(each_year[year]['average_y_20'])
#    average_y_60 = np.array(each_year[year]['average_y_60'])
#    
#    
#    average_y_5_20 = np.array(each_year[year]['average_y_5_20'])    
#    average_y_5_60 = np.array(each_year[year]['average_y_5_60'])    
#    average_y_20_60 = np.array(each_year[year]['average_y_20_60'])  
#    
#    rsi = np.array(each_year[year]['rsi'])
#    
    
    
    
    converted_dates = datestr2num([ datetime.strptime(day, '%Y-%m-%d').strftime('%m/%d/%Y') for day in data_set['x_values'] ]) 


    x = converted_dates

    

    
    f = plt.figure()
    f.set_figheight(20)
    f.set_figwidth(40)





    plt.subplot(3, 1, 1)
    plt.title(data_set_id)
    
    plt.plot_date( x, data_set['y_values']          , fmt='-', marker = ' ' , label='Value')
    plt.plot_date( x, data_set['best_fit_line']     , fmt='-', marker = ' ' , label='linear')
    plt.plot_date( x, data_set['theta_fit_list_2']  , fmt='-', marker = ' ' , label='poly 2')      
    
    plt.legend()
    plt.grid(axis = 'both')
    plt.xticks(rotation=65, horizontalalignment='right')




    plt.subplot(3, 1, 2)
    plt.title(data_set_id + " - Value")


    
    plt.plot_date( x, data_set['y_values']       , fmt='-', marker = ' ' , label='Value')

    plt.plot_date(x, data_set['average_y_5']     , fmt='--', marker = ' ' , color='orange'  , label='Running average 5')
    plt.plot_date(x, data_set['average_y_20']    , fmt='--', marker = ' ' , color='green'   , label='Running average 20')
    plt.plot_date(x, data_set['average_y_60']    , fmt='--', marker = ' ' , color='red'     , label='Running average 60')
    
    
    
    
    plt.legend()
    plt.grid(axis = 'both')
    plt.xticks(rotation=65, horizontalalignment='right')









    plt.subplot(3, 1, 3)
    plt.title(data_set_id + " - Value")


    
    plt.plot_date( x, data_set['daily_percent_increase']       , fmt='-', marker = ' ' , label='Value')

    plt.plot_date(x, data_set['daily_percent_increase_ma_5']     , fmt='--', marker = ' ' , color='orange'  , label='Running average 5')
    plt.plot_date(x, data_set['daily_percent_increase_ma_20']    , fmt='--', marker = ' ' , color='green'   , label='Running average 20')
    plt.plot_date(x, data_set['daily_percent_increase_ma_60']    , fmt='--', marker = ' ' , color='red'     , label='Running average 60')
    plt.plot_date(x, data_set['daily_percent_increase_ma_250']    , fmt='--', marker = ' ' , color='purple'     , label='Running average 60')
        
    
    
    
    plt.legend()
    plt.grid(axis = 'both')
    plt.xticks(rotation=65, horizontalalignment='right')



    filename = data_set_id + "/value-ma.png"









    plt.savefig(filename)
    plt.clf()
    


def gen_ma_macd_rsi_chart(data_set_id, data_set):


    converted_dates = datestr2num([ datetime.strptime(day, '%Y-%m-%d').strftime('%m/%d/%Y') for day in data_set['x_values'] ]) 


    x = converted_dates

   
   
   
    plt.subplot(3, 1, 1)
    plt.title(data_set_id + " - MA")


    plt.grid(axis = 'both')
    plt.xticks(rotation=65, horizontalalignment='right')


    plt.plot_date(x, data_set['average_y_5']       , fmt='--', marker = ' ' , color='orange'      , label='Running average 5')
    plt.plot_date(x, data_set['average_y_20']      , fmt='--', marker = ' ' , color='green'       , label='Running average 20')
    plt.plot_date(x, data_set['average_y_60']      , fmt='--', marker = ' ' , color='red'         , label='Running average 60')

    plt.legend()




    plt.subplot(3, 1, 2)
    plt.title(data_set_id + " - MACD")


    plt.grid(axis = 'both')
    plt.xticks(rotation=65, horizontalalignment='right')

    plt.plot_date(x, data_set['average_y_5_20']    , fmt='--', marker = ' ' , color='green'     , label='Running average 5 - 20')
#    plt.plot_date(x, data_set['average_y_5_60']    , fmt='--', marker = ' ' , color='orange'    , label='Running average 5 - 60')
    plt.plot_date(x, data_set['average_y_20_60']   , fmt='--', marker = ' ' , color='red'       , label='Running average 20 - 60')

    plt.legend()






    plt.subplot(3, 1, 3)
    plt.title(data_set_id + " - RSI")


    upper_limit = 80
    lower_limit = 40

    #plt.figure(figsize=(10, 6))
    plt.grid(axis = 'both')
    plt.xticks(rotation=65, horizontalalignment='right')
    
    plt.plot_date(x, data_set['rsi'], fmt='-', marker = ' ', label='rsi' )
    
    plt.axhline(y=upper_limit, color='r', linestyle='--', label='Overbought (80)')
    plt.axhline(y=lower_limit, color='g', linestyle='--', label='Oversold (40)')


    plt.legend()


  
    filename = data_set_id + "/ma-macd.png"

    plt.savefig(filename)
    plt.clf()

    
    
    
    
    
    
# Initialize the ArgumentParser
parser = argparse.ArgumentParser(description="A program to demonstrate optional arguments.")

# Define optional arguments
#parser.add_argument("-s", "--short", help="This is a short option")
parser.add_argument("--ytd",    action='store_true', help="Run current YTD calculations")
parser.add_argument("--all",    action='store_true', help="Run all time calculations")
parser.add_argument("--years",  action='store_true', help="Run all past years calculations")

# Parse the arguments
args = parser.parse_args()

    
    
    
    





all_time = gen_all_time('P1.csv')
all_time.update(gen_best_fit(all_time))
    


each_year = gen_each_year(all_time)


all_time_high = gen_all_time_high(all_time)


time_frame_stats = gen_time_frame_stats(all_time)



print("high low cycle")
print()
print("------------+------------------+------------------+--------------+--------+----------+")
print("        date|     all time high|      all time low|    difference|    days|   percent|")
print("------------+------------------+------------------+--------------+--------+----------+")


for item in all_time_high :


    print('{0:>12s}|  {1:>16.2f}|  {2:>16.2f}|  {3:>12.2f}|  {4:6}|  {5:>8.2f}|'.format(
    
        item['date'] , 
        item['cycle_high'] , 
        item['cycle_low'] , 
        item['difference'] , 
        item['days'] , 
        item['percent']
    
    ))




print()
print()
print("yearly stats")
print()
print("------------+------------------+------------------+--------------------+-------------+")
print("        year|             slope|                r2|          % increase|   total days|")
print("------------+------------------+------------------+--------------------+-------------+")


for year in each_year:
    
    
    each_year[year].update(gen_best_fit(each_year[year]))
    


    print('{0:>12s}|  {1:>16.2f}|  {2:>16.3f}|  {3:>18.2f}|  {4:11}|'.format(
    
        year, 
        each_year[year]['slope'], 
        each_year[year]['r2'],
        each_year[year]['percent_increase'],
        len(each_year[year]['y_values'])
    ))
  



print()
print()
print("time frame stats")
print()
print("------------+------------------+------------------+--------------------+-------------+")
print("  time frame|             slope|                r2|          % increase|      daily %|")
print("------------+------------------+------------------+--------------------+-------------+")




for time_frame in time_frame_stats: 

    
    print('{0:>12s}|  {1:>16.2f}|  {2:>16.3f}|  {3:>18.2f}|  {4:11.2f}|'.format(
    
        time_frame, 
        time_frame_stats[time_frame]['slope'], 
        time_frame_stats[time_frame]['r2'],
        time_frame_stats[time_frame]['percent_increase'],
        time_frame_stats[time_frame]['daily_percent_increase']
    ))
        









print()
print("generating charts")

print("all_time")

if not os.path.exists("all_time"):
    os.makedirs("all_time")


gen_best_fit_ma_chart("all_time", all_time)

gen_ma_macd_rsi_chart("all_time", all_time)
    





for year in each_year:
    

    print(year)
    
    if not os.path.exists(year):
        os.makedirs(year)

    
    
    gen_best_fit_ma_chart(year, each_year[year])

    gen_ma_macd_rsi_chart(year, each_year[year])
    
    
   




    
    

#
# Generate pdf file 
#



pdf = fpdf(orientation="P", unit="mm", format="A4")
 
print()
print("pdf report")

pdf.add_page()
pdf.set_font("helvetica", "B", 20)

pdf.cell(0, 18, "Report" , 1 , align='C')
pdf.ln()

pdf.set_font("helvetica", "B", 12)
pdf.cell(0, 10, "Portfolio Date " , align='L' )
pdf.cell(0, 10, "Generated " ,align='R' )

pdf.ln()

pdf.set_font("helvetica", "", 12)
pdf.cell(0, 10, all_time['x_values'][-1] , align='L' )
pdf.cell(0, 10, time.strftime("%Y-%m-%d %H:%M") , align='R' )

pdf.ln(20)


pdf.set_font("helvetica", "B", 12)
pdf.cell(30, 10, "Year" )
pdf.cell(30, 10, "Linear slope" )
pdf.cell(30, 10, "R squared" )

pdf.ln()

for year in each_year:
    pdf.set_font("helvetica", "", 12)
    pdf.cell(30, 10, year )
    pdf.cell(30, 10, str(each_year[year]['slope']) )
    pdf.cell(30, 10, str(each_year[year]['r2']) )
    
    pdf.ln()









print("pdf charts")
print("all_time")


pdf.add_page()
pdf.image('all_time/value-ma.png', w = 200 , h = 250)    


pdf.add_page()
pdf.image( 'all_time/ma-macd.png', w = 200 , h = 250)    



for year in each_year:

    print(year)
    pdf.add_page()

    pdf.set_font("helvetica", "B", 20)
    pdf.cell(0, 18, year + " Report" , 1 , align='C')
    pdf.ln()

    pdf.add_page()
    pdf.image(year + '/value-ma.png', w = 200 , h = 250)    


    pdf.add_page()
    pdf.image(year + '/ma-macd.png', w = 200 , h = 250)    





pdf.output("report " + time.strftime("%Y-%m-%d %H-%M") + ".pdf")
pdf.output("report.pdf")
