import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import ta
import pandas as pd
from  fpdf import FPDF as fpdf
import time



def moving_average(window, inputValue):
    
    average_y = []

    for ind in range(len(inputValue) - window + 1):
        average_y.append(np.mean(inputValue[ind:ind+window]))


    for ind in range(window - 1):
        average_y.insert(0, np.nan)


    return average_y






x_values = []
y_values = []

each_year = {}





with open('P1.csv', newline='') as csvfile:

    reader = csv.reader(csvfile)

    for row in reader:
        if row[1] != "" and row[1] != "Date":

            #print(row[1].split(" ")[0],row[2])
            x_values.append(row[1].split(" ")[0])
            y_values.append(float(row[2]))
            



            if row[1].split(" ")[0].split("-")[0] not in each_year:
                each_year[row[1].split(" ")[0].split("-")[0]] = {}

                each_year[row[1].split(" ")[0].split("-")[0]]['x_values'] = [row[1].split(" ")[0]]
                each_year[row[1].split(" ")[0].split("-")[0]]['y_values'] = [float(row[2])]
                each_year[row[1].split(" ")[0].split("-")[0]]['average_y_5'] = []
                each_year[row[1].split(" ")[0].split("-")[0]]['average_y_20'] = []
                each_year[row[1].split(" ")[0].split("-")[0]]['average_y_60'] = []
                
                each_year[row[1].split(" ")[0].split("-")[0]]['average_y_5_20'] = []
                each_year[row[1].split(" ")[0].split("-")[0]]['average_y_5_60'] = []                
                
                each_year[row[1].split(" ")[0].split("-")[0]]['rsi'] = []   

                
            else:
                each_year[row[1].split(" ")[0].split("-")[0]]['x_values'].append(row[1].split(" ")[0])
                each_year[row[1].split(" ")[0].split("-")[0]]['y_values'].append(float(row[2]))
                
                
                
                
            
    average_y_5 = moving_average(5 ,y_values)
    average_y_20 = moving_average(20 ,y_values)
    average_y_60 = moving_average(60 ,y_values)



    df = pd.DataFrame({'Actual': y_values})

    rsi = ta.momentum.RSIIndicator(df['Actual']).rsi()



    for i in range(len(x_values)) :
        year =x_values[i].split(" ")[0].split("-")[0]
        
        
        each_year[year]['average_y_5'].append(average_y_5[i])
        each_year[year]['average_y_20'].append(average_y_20[i])
        each_year[year]['average_y_60'].append(average_y_60[i])
        
        each_year[year]['average_y_5_20'].append(average_y_5[i] - average_y_20[i])
        each_year[year]['average_y_5_60'].append(average_y_5[i] - average_y_60[i])
        each_year[year]['rsi'].append(rsi[i])


        


for year in each_year:

    print(year)
    
    if not os.path.exists(year):
        os.makedirs(year)


    
    
    x = np.array(each_year[year]['x_values'])
    y = np.array(each_year[year]['y_values'])

    
    average_y_5 = np.array(each_year[year]['average_y_5'])
    average_y_20 = np.array(each_year[year]['average_y_20'])
    average_y_60 = np.array(each_year[year]['average_y_60'])
    
    
    average_y_5_20 = np.array(each_year[year]['average_y_5_20'])    
    average_y_5_60 = np.array(each_year[year]['average_y_5_60'])    
    
    rsi = np.array(each_year[year]['rsi'])
    
    
    

    m, b = np.polyfit(np.array(range(len(x))), y, 1)
    
    theta_2 = np.polyfit(np.array(range(len(x))), y, 2)
    theta_3 = np.polyfit(np.array(range(len(x))), y, 3)
    
    
    
    each_year[year]['linear_slop'] = round(m, 3)
 
    
    best_fit_list = []
    theta_fit_list_2 =[]
    theta_fit_list_3 =[]

    
    for i in range(len(each_year[year]['x_values'])):

        best_fit_list.append(m*i+b)
        z_2 = np.poly1d(theta_2)
        z_3 = np.poly1d(theta_3)
        
        theta_fit_list_2.append(z_2(i))
        theta_fit_list_3.append(z_3(i))
    
    best_fit = np.array(best_fit_list)
    
    theta_fit_2 = np.array(theta_fit_list_2)
    theta_fit_3 = np.array(theta_fit_list_3)
    
    
    
    
    
    
    f = plt.figure()
    f.set_figheight(20)
    f.set_figwidth(40)

#
# each year chart 1
#




    plt.subplot(2, 1, 1)
    plt.title(year)
    
    plt.plot( x, y, label='Value')
    plt.plot( x, best_fit, label='linear')
    plt.plot( x, theta_fit_2, label='poly 2')   
    plt.plot( x, theta_fit_3, label='poly 3')       
    
    plt.legend()
    plt.grid(axis = 'y')
    plt.xticks(rotation=65)







    plt.subplot(2, 1, 2)
    plt.title(year + " - Value")


    
    plt.plot( x, y, label='Value')

    plt.plot(x, average_y_5 , linestyle = 'dashed', color='orange' , label='Running average 5')
    plt.plot(x, average_y_20, linestyle = 'dashed', color='green' , label='Running average 20')
    plt.plot(x, average_y_60, linestyle = 'dashed', color='red' , label='Running average 60')
    
    
    
    
    
    plt.legend()
    plt.grid(axis = 'y')
    plt.xticks(rotation=65)



    filename = year + "/value-ma.png"

    plt.savefig(filename)
    plt.clf()
    
    

#
# each year chart 2
#




    plt.subplot(3, 1, 1)
    plt.title(year + " - MA")


    plt.grid(axis = 'y')
    plt.xticks([])


    plt.plot(x, average_y_5 , linestyle = 'dashed', color='orange' , label='Running average 5')
    plt.plot(x, average_y_20, linestyle = 'dashed', color='green' , label='Running average 20')
    plt.plot(x, average_y_60, linestyle = 'dashed', color='red' , label='Running average 60')

    plt.legend()




    plt.subplot(3, 1, 2)
    plt.title(year + " - MACD")


    plt.grid(axis = 'y')
    plt.xticks([])

    plt.plot(x, average_y_5_20, linestyle = 'dashed', color='green' , label='Running average 5 - 20')
    plt.plot(x, average_y_5_60, linestyle = 'dashed', color='red' , label='Running average 5 - 60')

    plt.legend()








    plt.subplot(3, 1, 3)
    plt.title(year + " - RSI")


    upper_limit = 70
    lower_limit = 30

    #plt.figure(figsize=(10, 6))
    plt.grid(axis = 'y')
    plt.xticks(rotation=65)
    
    plt.plot(x, rsi)
    
    plt.axhline(y=upper_limit, color='r', linestyle='--', label='Overbought (70)')
    plt.axhline(y=lower_limit, color='g', linestyle='--', label='Oversold (30)')





  
    filename = year + "/ma-macd.png"

    plt.savefig(filename)
    plt.clf()

    
    
    
    
    



    
    
    
    
    
#
# create all time charts 
#    
    







x_points = np.array(x_values)
y_points = np.array(y_values)


average_y_5 = moving_average(5 ,y_points)
average_y_20 = moving_average(20 ,y_points)
average_y_60 = moving_average(60 ,y_points)






f = plt.figure()
f.set_figheight(10)
f.set_figwidth(25)

plt.subplot(2, 1, 1)
plt.title("Value")
plt.plot( x_points, y_points)


plt.plot(x_points, average_y_5 , linestyle = 'dashed', label='Running average 5')
plt.plot(x_points, average_y_20, linestyle = 'dashed', label='Running average 20')
plt.plot(x_points, average_y_60, linestyle = 'dashed', label='Running average 60')


plt.subplot(2, 1, 2)
plt.title("MA")
plt.plot(x_points, average_y_5 , linestyle = 'dashed', label='Running average 5')
plt.plot(x_points, average_y_20, linestyle = 'dashed', label='Running average 20')
plt.plot(x_points, average_y_60, linestyle = 'dashed', label='Running average 60')



plt.legend()

plt.grid(axis = 'y')
plt.xticks(rotation=65)


plt.savefig('chart1.png')
plt.clf()









#
# Generate pdf file 
#



pdf = fpdf(orientation="P", unit="mm", format="A4")
 



pdf.add_page()
pdf.set_font("helvetica", "B", 16)

pdf.cell(40, 10, "Report" , 1 , align='C')
pdf.ln()

pdf.set_font("helvetica", "B", 12)
pdf.cell(40, 10, "generated " + time.strftime("%Y-%m-%d %H:%M") )
pdf.ln()
pdf.ln()
pdf.ln()

for year in each_year:
    pdf.set_font("helvetica", "B", 12)
    pdf.cell(40, 10, year )
    pdf.ln()
    pdf.cell(40, 10, str(each_year[year]['linear_slop']) )
    pdf.ln()

    pdf.ln()










pdf.add_page()

pdf.image('chart1.png', w = 200 , h = 250)    





for year in each_year:
    print(year)
    pdf.add_page()
    pdf.cell(40, 10, year + " Report" , 1 , align='C')
    pdf.ln()

    pdf.add_page()
    pdf.image(year + '/value-ma.png', w = 200 , h = 250)    


    pdf.add_page()
    pdf.image(year + '/ma-macd.png', w = 200 , h = 250)    


















pdf.output("report " + time.strftime("%Y-%m-%d %H-%M") + ".pdf")
pdf.output("report.pdf")