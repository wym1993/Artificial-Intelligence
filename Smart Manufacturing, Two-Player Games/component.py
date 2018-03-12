from widgetNode import WidgetNode
from copy import deepcopy

cities = ['A', 'B', 'C', 'D', 'E']

class Status(object):

	def __init__(self, city, widgets):
		self.city = city
		self.widgets = widgets
		self.prev_status = []

	def __eq__(self, other):
		return self.city == other.city and self.check_left() == other.check_left()

	def check_left(self):
		return [widget.get_left() for widget in self.widgets]

	def generate_nxt_status(self):
		for city in cities:
			if city != self.city:
				nxt = deepcopy(self.widgets)
				for widget in nxt:
					widget.add_city(city)
				yield Status(city, nxt)

class Layer(object):

	def __init__(self, status):
		self.status_list = status

	def find_status_from_goal(self, target, city):
		for status in self.status_list:
			if status.check_left() == target and status.city == city:
				return status
		return None

	def get_nxt_layer(self):
		nxt_cands = []
		for status in self.status_list:
			for nxt_status in status.generate_nxt_status():
				should_add = True
				for cand in nxt_cands:
					if cand == nxt_status:
						should_add = False
						cand.prev_status.append(status)
				if should_add:
					nxt_status.prev_status.append(status)
					nxt_cands.append(nxt_status)
		return nxt_cands



