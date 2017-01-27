#define the class Station, which has an id, a location list of coordinates, and a temperature
#initialized with id, longitude, latitude, altitude, temperature. long/lat/alt can go into the location list
class Station:

	def __init__(self, id, longitude, latitude, altitude, temperature):
		self.id = id #syntax highlighting irrelevant here (id is a built-in function but here it refers to the argument)
		self.loc = [0, 0]
		self.loc[0] = longitude
		self.loc[1] = latitude
		#self.loc[2] = altitude
		self.temperature = temperature