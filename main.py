import csv
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




    with open(file, newline='') as csvfile:

    #with open('Net Worth.csv', newline='') as csvfile:

        reader = csv.reader(csvfile)

        for row in reader:
            if row[1] != "" and row[1] != "Date":


                all_time['x_values'].append(row[1].split(" ")[0])
                all_time['y_values'].append(float(row[2]))
               




        all_time['average_y_5'] = moving_average(5 ,all_time['y_values'])
        all_time['average_y_20'] = moving_average(20 ,all_time['y_values'])
        all_time['average_y_60'] = moving_average(60 ,all_time['y_values'])



        df = pd.DataFrame({'Actual': all_time['y_values']})

        all_time['rsi'] = list(ta.momentum.RSIIndicator(df['Actual']).rsi())
        
        
        
        
    return all_time



def gen_each_year(all_time):

    each_year = {}

    for index, value in enumerate(all_time['y_values']) :

        year =x_values[index].split(" ")[0].split("-")[0]
        
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


        each_year[year]['x_values'].append(x_values[index])
        each_year[year]['y_values'].append(value)
        
        each_year[year]['average_y_5'].append(average_y_5[index])
        each_year[year]['average_y_20'].append(average_y_20[index])
        each_year[year]['average_y_60'].append(average_y_60[index])
        
        each_year[year]['average_y_5_20'].append(average_y_5[index] - average_y_20[index])
        each_year[year]['average_y_5_60'].append(average_y_5[index] - average_y_60[index])
        each_year[year]['average_y_20_60'].append(average_y_20[index] - average_y_60[index])    
        
        each_year[year]['rsi'].append(rsi[index])


    return each_year



def gen_all_time_high(all_time):

    all_time_high = []
    
    cycle_high = 0
    cycle_low = y_values[0]

 
    for index, value in enumerate(y_values) :

        if value > cycle_high : 
            
            if cycle_high != cycle_low and index != 0 and cycle_low_index - cycle_high_index > 2:
                item = {}
                
                
                item['date'] = x_values[cycle_high_index] 
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















all_time = gen_all_time('P1.csv')




x_values = all_time['x_values']
y_values = all_time['y_values']


average_y_5 = all_time['average_y_5']
average_y_20 = all_time['average_y_20']
average_y_60 = all_time['average_y_60']

rsi = all_time['rsi']




each_year = gen_each_year(all_time)






all_time_high = gen_all_time_high(all_time)




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
print("        year|             slope|                r2|    percent increase|   total days|")
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
        



    if year == list(each_year.items())[-1][0] : 
    
    

        print()
        print("last 12m")
        print((y_values[-1] - y_values[-250])/y_values[-250])
        print(250)        
    













for year in each_year:
    
   

    average_y_5 = each_year[year]['average_y_5']
    average_y_20 = each_year[year]['average_y_20']
    average_y_60 = each_year[year]['average_y_60']
    
    
    average_y_5_20 = each_year[year]['average_y_5_20']  
    average_y_5_60 = each_year[year]['average_y_5_60'] 
    average_y_20_60 = each_year[year]['average_y_20_60']
    
    rsi = each_year[year]['rsi']
    
    
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
    
    
    
    converted_dates = datestr2num([ datetime.strptime(day, '%Y-%m-%d').strftime('%m/%d/%Y') for day in each_year[year]['x_values'] ]) 


    x = converted_dates
    y = each_year[year]['y_values']

    
    
    best_fit = each_year[year]['best_fit_line']
    
    theta_fit_2 = each_year[year]['theta_fit_list_2']

    
    
    if not os.path.exists(year):
        os.makedirs(year)

    
    
    
    f = plt.figure()
    f.set_figheight(20)
    f.set_figwidth(40)

#
# each year chart 1
# value and ma
#




    plt.subplot(2, 1, 1)
    plt.title(year)
    
    plt.plot_date( x, y, fmt='-', marker = ' ' , label='Value')
    plt.plot_date( x, best_fit, fmt='-', marker = ' ' , label='linear')
    plt.plot_date( x, theta_fit_2, fmt='-', marker = ' ' , label='poly 2')      
    
    plt.legend()
    plt.grid(axis = 'both')
    plt.xticks(rotation=65, horizontalalignment='right')







    plt.subplot(2, 1, 2)
    plt.title(year + " - Value")


    
    plt.plot_date( x, y, fmt='-', marker = ' ' , label='Value')

    plt.plot_date(x, average_y_5 , fmt='--', marker = ' ' , color='orange' , label='Running average 5')
    plt.plot_date(x, average_y_20, fmt='--', marker = ' ' , color='green' , label='Running average 20')
    plt.plot_date(x, average_y_60, fmt='--', marker = ' ' , color='red' , label='Running average 60')
    
    
    
    
    
    plt.legend()
    plt.grid(axis = 'both')
    plt.xticks(rotation=65, horizontalalignment='right')



    filename = year + "/value-ma.png"

    plt.savefig(filename)
    plt.clf()
    
    

#
# each year chart 2
# ma , macd , rsi
#




    plt.subplot(3, 1, 1)
    plt.title(year + " - MA")


    plt.grid(axis = 'both')
    plt.xticks(rotation=65, horizontalalignment='right')


    plt.plot_date(x, average_y_5 , fmt='--', marker = ' ' , color='orange' , label='Running average 5')
    plt.plot_date(x, average_y_20, fmt='--', marker = ' ' , color='green' , label='Running average 20')
    plt.plot_date(x, average_y_60, fmt='--', marker = ' ' , color='red' , label='Running average 60')

    plt.legend()




    plt.subplot(3, 1, 2)
    plt.title(year + " - MACD")


    plt.grid(axis = 'both')
    plt.xticks(rotation=65, horizontalalignment='right')

    plt.plot_date(x, average_y_5_20, fmt='--', marker = ' ' , color='green' , label='Running average 5 - 20')
#    plt.plot_date(x, average_y_5_60, fmt='--', marker = ' ' , color='orange' , label='Running average 5 - 60')
    plt.plot_date(x, average_y_20_60, fmt='--', marker = ' ' , color='red' , label='Running average 20 - 60')

    plt.legend()








    plt.subplot(3, 1, 3)
    plt.title(year + " - RSI")


    upper_limit = 70
    lower_limit = 30

    #plt.figure(figsize=(10, 6))
    plt.grid(axis = 'both')
    plt.xticks(rotation=65, horizontalalignment='right')
    
    plt.plot_date(x, rsi, fmt='-', marker = ' ' )
    
    plt.axhline(y=upper_limit, color='r', linestyle='--', label='Overbought (70)')
    plt.axhline(y=lower_limit, color='g', linestyle='--', label='Oversold (30)')





  
    filename = year + "/ma-macd.png"

    plt.savefig(filename)
    plt.clf()

    
    
    
    
    



    
    
    
    
    
#
# create all time charts 
#    
    




converted_dates = datestr2num([ datetime.strptime(day, '%Y-%m-%d').strftime('%m/%d/%Y') for day in x_values ]) 




x_points = np.array(converted_dates)
y_points = np.array(y_values)


average_y_5 = moving_average(5 ,y_points)
average_y_20 = moving_average(20 ,y_points)
average_y_60 = moving_average(60 ,y_points)






f = plt.figure()
f.set_figheight(10)
f.set_figwidth(25)

plt.subplot(2, 1, 1)
plt.title("Value")



plt.plot_date( x_points, y_points , fmt='-', marker = ' ' )

plt.plot_date(x_points, average_y_5 , fmt='--', marker = ' ' , label='Running average 5')
plt.plot_date(x_points, average_y_20, fmt='--', marker = ' ' , label='Running average 20')
plt.plot_date(x_points, average_y_60, fmt='--', marker = ' ' , label='Running average 60')

plt.legend()
plt.xticks(rotation=65, horizontalalignment='right')
plt.grid(axis = 'both')

plt.subplot(2, 1, 2)
plt.title("MA")
plt.plot_date(x_points, average_y_5 , fmt='--', marker = ' ' , label='Running average 5')
plt.plot_date(x_points, average_y_20, fmt='--', marker = ' ' , label='Running average 20')
plt.plot_date(x_points, average_y_60, fmt='--', marker = ' ' , label='Running average 60')



plt.legend()

plt.grid(axis = 'both')
plt.xticks(rotation=65, horizontalalignment='right')


plt.savefig('chart1.png')
plt.clf()









#
# Generate pdf file 
#



pdf = fpdf(orientation="P", unit="mm", format="A4")
 



pdf.add_page()
pdf.set_font("helvetica", "B", 20)

pdf.cell(0, 18, "Report" , 1 , align='C')
pdf.ln()

pdf.set_font("helvetica", "B", 12)
pdf.cell(0, 10, "Portfolio Date " , align='L' )
pdf.cell(0, 10, "Generated " ,align='R' )

pdf.ln()

pdf.set_font("helvetica", "", 12)
pdf.cell(0, 10, x_values[-1] , align='L' )
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











pdf.add_page()

pdf.image('chart1.png', w = 200 , h = 250)    





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
