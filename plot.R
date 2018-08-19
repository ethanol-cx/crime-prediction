library(gdata)
print("hello world")

data = read.csv(file='C:/Users/Chuan/OneDrive - University of Southern California/Summer 2018/Crime Prediction/la_crime_forecasting/crimeplot.csv', sep='\t')
index(data) <- as.Date(as.character(index(pricez)), format="%m%d/%y")


plot(data)

abline(reg=lm(data[1]~data[0]))