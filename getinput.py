import xlrd
from station import Station

"""
Functions for reading input
"""

#returns list of temperatures for a day; can be modified to return mean temperatures, list of temperatures, etc.
def getTemperatures(second_sheet, day):
	temperatures = second_sheet.row_values(day)
	temperatures.pop(0) #first row labels
	return temperatures

#returns a list of Stations objects from the excel file
def getStations(day):
	book = xlrd.open_workbook("data.xls")
	first_sheet = book.sheet_by_index(0)
	second_sheet = book.sheet_by_index(1)
	
	stations = []
	temperatures = getTemperatures(second_sheet, day)
	st_count = 1

	while (st_count < len(first_sheet.col_values(0))):
		curr_id = first_sheet.col_values(0)[st_count]
		curr_latitude = first_sheet.col_values(1)[st_count]
		curr_longitude = first_sheet.col_values(2)[st_count]
		curr_altitude = first_sheet.col_values(3)[st_count]
		curr_temp = temperatures[st_count - 1]

		curr_station = Station(curr_id, curr_longitude, curr_latitude, curr_altitude, curr_temp)
		stations.append(curr_station)
		st_count += 1

	return stations

#print data to ensure it has been read correctly
def printData(stations):
	for i in range(len(stations)):
		print stations[i].id, stations[i].loc[0], stations[i].loc[1], stations[i].temperature