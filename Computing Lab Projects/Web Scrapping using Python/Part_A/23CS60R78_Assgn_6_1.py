import requests
import sqlite3
import json
from bs4 import BeautifulSoup


def getData(url):
    response = requests.get(url)
    #convert to text string and return 
    return response.text

def convertJson(data):
    return json.loads(data)

def createDatabaseConnect(dbName):
	con = sqlite3.connect(dbName)
	cur = con.cursor()
	return con,cur

###########################################################################

# We will call https://cse.iitkgp.ac.in/~mainack/computing-lab/web-scraping/json.php
# Lets see what it returns
url = 'http://api.openweathermap.org/data/2.5/weather?lat=22.5726723&lon=88.3638815&appid=418ed97630b77a97a739bc813e87af27'
returnedData = getData(url)
#print(returnedData)
## it will print {"a":"Apple","b":"Ball","c":"Cat"}

## This is structured data which can be converted into json format. 
## lets do that 
jsonData = convertJson(returnedData)
## Now jsonData contains a python dict
name=jsonData['name']
temp=jsonData['main']['temp']
description=jsonData['weather'][0]['description']
humidity=jsonData['main']['humidity']
wind_speed=jsonData['wind']['speed']
# print(name)
# print(temp)
# print(description)
# print(humidity)
# print(wind_speed)


###########################################################################
## Lastly lets store the data in a database

## create the database or if it already exisit connect to it

dbName = "Weather.db"
con,cursor = createDatabaseConnect(dbName)

# ## Now you can create Table and insert/select records from there
# ## Lets create a Table "example" with three columns a, b and c to insert the structured data 
# ## we fecthed earlier


query = "CREATE TABLE IF NOT EXISTS city_weather(City, Temperature, Description, Humidity, WindSpeed)"
cursor.execute(query)

query = "INSERT INTO city_weather VALUES ('%s', '%s', '%s', '%s', '%s')"%(name, temp, description, humidity, wind_speed)
cursor.execute(query)
con.commit()




#2
url = 'http://api.openweathermap.org/data/2.5/weather?lat=51.509865&lon=-0.118092&appid=418ed97630b77a97a739bc813e87af27'
returnedData = getData(url)
jsonData = convertJson(returnedData)
name=jsonData['name']
temp=jsonData['main']['temp']
description=jsonData['weather'][0]['description']
humidity=jsonData['main']['humidity']
wind_speed=jsonData['wind']['speed']


dbName = "Weather.db"
con,cursor = createDatabaseConnect(dbName)
query = "CREATE TABLE IF NOT EXISTS city_weather(City, Temperature, Description, Humidity, WindSpeed)"
cursor.execute(query)


query = "INSERT INTO city_weather VALUES ('%s', '%s', '%s', '%s', '%s')"%(name, temp, description, humidity, wind_speed)
cursor.execute(query)
con.commit()


#3
url = 'http://api.openweathermap.org/data/2.5/weather?lat=55.751244&lon=37.618423&appid=418ed97630b77a97a739bc813e87af27'
returnedData = getData(url)
jsonData = convertJson(returnedData)
name=jsonData['name']
temp=jsonData['main']['temp']
description=jsonData['weather'][0]['description']
humidity=jsonData['main']['humidity']
wind_speed=jsonData['wind']['speed']


dbName = "Weather.db"
con,cursor = createDatabaseConnect(dbName)
query = "CREATE TABLE IF NOT EXISTS city_weather(City, Temperature, Description, Humidity, WindSpeed)"
cursor.execute(query)
query = "INSERT INTO city_weather VALUES ('%s', '%s', '%s', '%s', '%s')"%(name, temp, description, humidity, wind_speed)
cursor.execute(query)
con.commit()


#4
url = 'http://api.openweathermap.org/data/2.5/weather?lat=38.889805&lon=-77.009056&appid=418ed97630b77a97a739bc813e87af27'
returnedData = getData(url)
jsonData = convertJson(returnedData)
name=jsonData['name']
temp=jsonData['main']['temp']
description=jsonData['weather'][0]['description']
humidity=jsonData['main']['humidity']
wind_speed=jsonData['wind']['speed']


dbName = "Weather.db"
con,cursor = createDatabaseConnect(dbName)
query = "CREATE TABLE IF NOT EXISTS city_weather(City, Temperature, Description, Humidity, WindSpeed)"
cursor.execute(query)
query = "INSERT INTO city_weather VALUES ('%s', '%s', '%s', '%s', '%s')"%(name, temp, description, humidity, wind_speed)
cursor.execute(query)
con.commit()


#5
url = 'http://api.openweathermap.org/data/2.5/weather?lat=55.860916&lon=-4.251433&appid=418ed97630b77a97a739bc813e87af27'
returnedData = getData(url)
jsonData = convertJson(returnedData)
name=jsonData['name']
temp=jsonData['main']['temp']
description=jsonData['weather'][0]['description']
humidity=jsonData['main']['humidity']
wind_speed=jsonData['wind']['speed']

dbName = "Weather.db"
con,cursor = createDatabaseConnect(dbName)
query = "CREATE TABLE IF NOT EXISTS city_weather(City, Temperature, Description, Humidity, WindSpeed)"
cursor.execute(query)
query = "INSERT INTO city_weather VALUES ('%s', '%s', '%s', '%s', '%s')"%(name, temp, description, humidity, wind_speed)
cursor.execute(query)
con.commit()
query = "SELECT * from city_weather"
result = cursor.execute(query)
for row in result:
    print(row)
cursor.close()







