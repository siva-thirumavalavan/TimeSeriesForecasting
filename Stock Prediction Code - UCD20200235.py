'''
This code was writted by Siva Thirumavalavan
ID: UCD20200235
'''


import pandas as pd
import re
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pylab as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings("ignore")
from tkinter import Tk
from tkinter.filedialog import askdirectory

path = askdirectory(title='Select Folder') #prompts the user to select directory and returns the path

tickersAvailable = pd.read_excel(path+'//TickerSymbols.xlsx',sheet_name='Basic View',usecols="A:C") #reads the excel file that has the list of tickers available

dateMap = pd.read_excel(path+'//dateMap.xlsx') #reads a dateMap file
dateMap['charDate']=list(map(lambda date: str(date.strftime('%Y-%m-%d')),dateMap.Date))


def tickerSearch(df):
    '''
    Prompts the user to enter the ticker keyword.
    Checks if the user's keyword is present in the master list. If not, prompts the user to type again
    returns list of Tickers from the master list matching the keyword
    '''
    searchKey = str(input("\nSearch for ticker symbol using keyword: ")).upper()
    filteredList = list(filter(lambda tickerSymbol: searchKey in tickerSymbol, df.Symbol))
    if (len(filteredList)==0):
        print("\nSorry. No results found.\nPlease try using a different keyword")
        filteredList = tickerSearch(df)
    return(filteredList)

def showMenu(optionList):
    '''
    gets a list of strings and displays it like a menu
    '''
    print("\nSelect your choice from the following")
    for i in range(1,len(optionList)+1):
        print(i,'.',optionList[i-1])

def invalidOption(optionList):
    '''
    if the choice entered is not present in the options, alerts the user and triggers series of functions enabling the user to select again
    '''
    print('\nInvalid option selected. Please try again')
    return(selectionIndexHandle(optionList))


def selectionIndexHandle(optionList):
    '''
    Prompts the user to enter thier choice based on the menu.
    triggers invalidOption() when an invalid option is entered
    receives the menu and returns the user selection from the menu
    '''
    showMenu(optionList)
    try:
        choice = int(input("Enter your choice: "))
        if (choice <= 0):
            selection = invalidOption(optionList)
        else:
            try:
                selection = optionList[choice-1]
            except IndexError as indError:
                selection = invalidOption(optionList)
    except ValueError as valError:
        selection = invalidOption(optionList)
    return(selection)


def tickerSelection(tickerList):
    '''
    triggers and receives the user selection from selectionIndexHandle()
    Promts the user to confirm their selection.
    Repeats the steps when the user wishes to reselect
    retunrs the ticker symbol once the user confirms selection
    '''
    tickerSelected = selectionIndexHandle(tickerList)
    print("\nYou have selected '",tickerSelected,"'")
    try:
        confrim = int(input('\nPress 1 to confirm. 0 to select again: '))
        if(confrim == 0):
            print('\n')
            tickerSelected = tickerSelection(tickerList)
    except ValueError as valError:
        print('\n Invalid entry')
        tickerSelected = tickerSelection(tickerList)
    return(tickerSelected)


def Search_for_Tickers():
    '''
    triggerd when user selects option 1 from the main menu
    triggers tickerSearch() enabling user to search for keyword and tickerSelection() to confrim the selection
    gives options to the users to download, analyse or predict on the selected Ticker
    triggers a function according to the user selection
    '''
    tickerSelected = tickerSelection(tickerSearch(tickersAvailable))
    print("\nWhat do you want to do with '",tickerSelected,"'?")
    globals()[selectionIndexHandle(['Download Ticker data','Analyse Ticker','Predict Stock Price']).replace(" ","_")](tickerSelected)


def tickerCheck():
    '''
    prompts the user to enter a ticker symbol. checks the symbol with master list.
    Prompts the user to enter again if teh symbol is not present in the master list
    '''

    ticker = str(input('\nEnter the Ticker symbol: ')).upper()
    if(ticker not in list(tickersAvailable.Symbol)):
        print('\nTicker not available. Please try a different Tricker')
        ticker = tickerCheck()
    return(ticker)

def getDate(query):
    '''
    promts the user to enter the 'query' date. checks the format of the date and promts the user to enter again if the format is invalid
    '''
    formattedDate  = input("\nEnter the " + query + " date in YYYY-MM-DD format ")
    if(formattedDate in list(dateMap.charDate)):
        if (query=='start'):
            queryDate = dateMap.charDate[dateMap[dateMap.charDate==formattedDate].index[0]]
        else:
            queryDate = dateMap.charDate[dateMap[dateMap.charDate==formattedDate].index[0]+1]
    else:
        print('\nIncorrect date format. Please re-enter')
        queryDate = getDate(query)
    return(queryDate)

def checkDates():
    '''
    promts the user to enter start and end period for the ticker data.
    Checks if the end date is older than start date.
    '''

    startDate =  getDate(str('start'))
    endDate = getDate(str('end'))
    if(startDate>endDate):
        print('\nStart date cannot be later than end date. Please re-enter')
        rerun = checkDates()
        startDate = rerun['startDate']
        endDate = rerun['endDate']
    return{'startDate':startDate,'endDate':endDate}


def downloadCreds(ticker):
    '''
    receives a ticker symbol, triggers checkDates() to get the start and end period for teh data and downloads it.
    If there is no data available for the given period, prompts the user to enter a different period
    '''
    if(ticker==None):
        ticker = tickerCheck()
    dates = checkDates()
    tickerdf = yf.download(ticker,start = str(dates['startDate']), end = str(dates['endDate']))

    if(tickerdf.shape[0]==0):
        print('The selected Ticker data is unavailable for the selected timeframe.\nPlease try a different combination')
        #removeTicker(ticker)
        #tickerdf = downloadCreds(tickerCheck())
    fileName = ticker+'_'+dates['startDate']+'_'+dates['endDate']+'.csv'
    return{'df':tickerdf,'file':fileName,'ticker':ticker,'start':dates['startDate'],'end':dates['endDate']}



def Download_Ticker_data(ticker = None):
    '''
    triggers downloadCreds() to download data from yfinance.
    prompts the user to enter a working directory and exports the data as csv
    '''
    tickerDownloaded = downloadCreds(ticker)
    outPath = askdirectory(title='Select Folder')
    #outPath = input('Enter the path to save the downloaded data: ')
    tickerDownloaded['df'].to_csv(outPath+'//'+tickerDownloaded['file'])
    print('Export Successful')


def runAnalysis(col):
    '''
    runs all the descriptive analysis on the column selected by the user
    '''
    col.dropna(inplace=True)
    print('\nWe have {} entries'.format(round(col.count(),2)))
    print('\nThe mean value is ',round(col.mean(),2))
    print('\nThe median value is ',round(col.median(),2))
    print('\nThe Standard deviation is {} and the variance is {}'.format(round(col.std(),2),round(col.std()**2,2)))
    print('\nThe range is ',round(col.max()-col.min(),2))
    print('\nThe Q1 is {}, Q2 is {}, Q3 is {} and the IQR is {}'.format(round(col.quantile(0.25),2),round(col.quantile(0.5),2),round(col.quantile(0.75),2),round(col.quantile(0.75)-col.quantile(0.25),2)))
    print('\nThe co-efficient of variation is ',round(col.std()/col.mean(),2))

def gotomainmenu(analyseCol=None,ticker=None,var=None):
    '''
    triggers the mainMenu() function when user selects go to main menu option from any window
    '''
    mainMenu()

def quit(analyseCol=None,ticker=None,var=None):
    '''
    Prints a thank you messgae
    '''
    print('{:*^100s}'.format("Happy Trading"))
    print('{:*^100s}'.format("No Price is too low for a bear or too high for a bull"))


def plottimeseries(analyseCol,ticker,var):
    '''
    triggerd by Analyse_Ticker().
    receives the column variable and plots raw time-series for the variable
    Gives user the option to view other graphs or quit the session.
    '''

    plt.plot(analyseCol)
    plt.title("Raw time series of {}'s {}".format(ticker,var))
    plt.show()
    print('\nDo you want to perform any further analysis from the following?')
    globals()[selectionIndexHandle(['Plot Moving Average','Plot Weighted Moving Average','Plot MACD','Go to Main Menu','Quit']).replace(" ","").lower()](analyseCol,ticker,var)

def getWindow(analyseCol):
    '''
    enables the user to enter the rolling window.
    alerts the user when the rolling window is greater than the sample points and prompts them to re-enter
    '''

    try:
        wdw = int(input("\nEnter the rolling window: "))
        if (wdw > len(analyseCol)):
            print("\n The specified window is greater than the sample data points. Please re-enter")
            wdw = getWindow(analyseCol)
    except ValueError as valError:
        print('Invalid entry. Please enter an integer')
        wdw = getWindow(analyseCol)
    return(wdw)

def plotmovingaverage(analyseCol,ticker,var):
    '''
    triggerd by Analyse_Ticker().
    receives the column variable and prompts the user to input the rolling window by triggerig getWindow()
    plots moving average for the variable
    Gives user the option to view other graphs or quit the session.
    '''
    wdw = getWindow(analyseCol)
    rolled = analyseCol.rolling(window = wdw).mean()
    plt.plot(rolled)
    plt.title("Moving average of {}'s {} with a rolling window of {}".format(ticker,var,wdw))
    plt.show()
    print('\nDo you want to perform any further analysis from the following?')
    globals()[selectionIndexHandle(['Plot Time Series','Plot Weighted Moving Average','Plot MACD','Go to Main Menu','Quit']).replace(" ","").lower()](analyseCol,ticker,var)

def plotweightedmovingaverage(analyseCol,ticker,var):
    '''
    triggerd by Analyse_Ticker().
    receives the column variable and prompts the user to input the rolling window by triggerig getWindow()
    plots weighted moving average for the variable
    Gives user the option to view other graphs or quit the session.
    '''
    wdw = getWindow(analyseCol)
    wma = analyseCol.ewm(halflife=wdw,min_periods=0,adjust=True).mean()
    plt.plot(wma)
    plt.title("Moving average of {}'s {} with a rolling window of {}".format(ticker,var,wdw))
    plt.show()
    print('\nDo you want to perform any further analysis from the following?')
    globals()[selectionIndexHandle(['Plot Time Series','Plot Moving Average','Plot MACD','Go to Main Menu','Quit']).replace(" ","").lower()](analyseCol,ticker,var)

def plotmacd(analyseCol,ticker,var):
    '''
    triggerd by Analyse_Ticker().
    receives the column variable and plots Moving Average Convergence Diverdence
    Gives user the option to view other graphs or quit the session.
    '''

    plt.plot(analyseCol,label = 'Observed', color = 'blue')
    plt.plot(analyseCol.ewm(halflife=len(analyseCol)/4,min_periods=0,adjust=True).mean()-analyseCol.ewm(halflife=len(analyseCol)/2,min_periods=0,adjust=True).mean(),label='MACD',color='red')
    plt.legend(loc='best')
    plt.title("MACD of {}'s {}".format(ticker,var))
    plt.show()
    print('\nDo you want to perform any further analysis from the following?')
    globals()[selectionIndexHandle(['Plot Time Series','Plot Moving Average','Plot Weighted Moving Average','Go to Main Menu','Quit']).replace(" ","").lower()](analyseCol,ticker,var)

def Analyse_Ticker(ticker=None):
    '''
    triggered when the user selects option 3 from the mainMenu or through search Menu
    gets ticker data from tickerDownloaded() and runs ARIMA forecast model on the Closing Price column
    prompts getPredDate() the user to give a forecast date and predicts the closing price for that date
    Gives user the option to view model graphs or quit the session.
    '''
    tickerDownloaded = downloadCreds(ticker)
    tickerdf = tickerDownloaded['df']
    ticker = tickerDownloaded['ticker']
    var = selectionIndexHandle(tickerdf.columns)
    analyseCol = tickerdf[var]
    if(var != 'Volume'):
        var = var + ' price'
    if(len(analyseCol>0)):
        runAnalysis(analyseCol)
        print('\nDo you want to perform any further analysis from the following?')
        globals()[selectionIndexHandle(['Plot Time series','Plot Moving Average','Plot Weighted Moving Average','Plot MACD']).replace(" ","").lower()](analyseCol,ticker,var)

    else:
        print('\nNo data available on Ticker for the specified timeframe')


def mainMenu():
    print('{:*^100s}'.format("Welcome"))
    Ticker = None
    options = ['Search for Tickers','Download Ticker data','Analyse Ticker','Predict Stock Price','quit']
    return(globals()[selectionIndexHandle(options).replace(" ","_")]())


from statsmodels.tsa.stattools import adfuller
def dickeyfullertestresults(model,testPrediction,dataset):
    '''
    calculates and displays the Dickey-Fuller test result for timeseries prediction
    gives user the option to quit or to see other model results
    '''
    dftest = adfuller(dataset['Close'],autolag = 'AIC')
    print("\n1. ADF : ", dftest[0])
    print("2. P-Value : ",dftest[1])
    print("3. Observations used for ADF Regression and critical values calculation : ",dftest[3])
    print("4. Critical Values :")
    for key, val in dftest[4].items():
        print("\t",key," : ",val)

    choiceList = ['Plot Actual vs predicted','See model summary','Quit']
    return(globals()[selectionIndexHandle(choiceList).replace(" ","").lower()](model,testPrediction,dataset))

def seemodelsummary(model,testPrediction,dataset):
    '''
    prints the model summary
    gives user the option to quit or to see other model results
    '''
    print(model.summary())
    choiceList = ['Plot Actual vs predicted','Dickey Fuller test results','Quit']
    return(globals()[selectionIndexHandle(choiceList).replace(" ","").lower()](model,testPrediction,dataset))


def plotactualvspredicted(model,testPrediction,dataset):
    '''
    plots the graph of actual vs predicted Price
    gives user the option to quit or to see other model results
    '''

    testPrediction.plot(legend =True, label = 'Predicted')
    datasetFiltered = dataset.loc[testPrediction.index]
    datasetFiltered['Close'].plot(legend=True, label = 'Actual')
    plt.title('Arima model on Closing price')
    plt.show()

    choiceList = ['See model summary','Dickey Fuller test results','Quit']
    return(globals()[selectionIndexHandle(choiceList).replace(" ","").lower()](model,testPrediction,dataset))

def getPredDate(endDate):
    '''
    enables the user to input the forecast date.
    Alerts when the user's date is older than end date and promts the user to enter a new date
    returns the date to Predict_Stock_Price() when conditions are satisfied
    '''
    pdate = getDate('Forecast')
    if(pdate < endDate):
        print('\nInvalid entry. Please select a forecast date later than end date')
        pdate = getPredDate(endDate)
    return(pdate)

def Predict_Stock_Price(ticker=None):
    '''
    triggered when the user selects option 4 from the mainMenu or through search Menu
    gets ticker data from downloadCreds() and runs ARIMA forecast model on the Closing Price column
    prompts getPredDate() the user to give a forecast date and predicts the closing price for that date
    Gives user the option to view model graphs or quit the session.
    '''

    tickerDownloaded = downloadCreds(ticker)
    startDate = tickerDownloaded['start']
    endDate = tickerDownloaded['end']
    tickerdf = tickerDownloaded['df']
    samplingSize = round(len(tickerdf)*0.8)
    train,test = tickerdf['Close'][0:samplingSize],tickerdf['Close'][samplingSize:len(tickerdf)]
    AR_model = ARIMA(train,order =(2,1,2))
    model = AR_model.fit()

    start = len(train)
    end = len(train) + len(test) - 1
    pdate = getPredDate(endDate)
    forecastDays = list(dateMap.charDate).index(pdate) - list(dateMap.charDate).index(endDate)

    testPrediction = model.predict(start = start, end = end,typ='levels')
    testPrediction.index=tickerdf.index[start:end+1]
    forecastedPrediction = model.predict(start = start, end = end + forecastDays,typ='levels')

    rmse = sqrt(mean_squared_error(testPrediction,test))
    print('\nThe RMSE value is',rmse)

    print('\nThe forecasted closing price is ',round(list(forecastedPrediction)[-1],2))
    choiceList = ['Plot Actual vs predicted','See model summary','Dickey Fuller test results','Quit']

    return(globals()[selectionIndexHandle(choiceList).replace(" ","").lower()](model,testPrediction,tickerdf))


mainMenu()
