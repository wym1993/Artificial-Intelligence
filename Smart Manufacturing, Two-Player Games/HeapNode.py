from widgetNode import WidgetNode
import sys

mile_matrix = [[0, 1064, 673, 1401, 277], \
				[1064, 0, 958, 1934, 337], \
				[673, 958, 0, 1001, 399], \
				[1401, 1934, 1001, 0, 387], \
				[277, 337, 399, 387, 0]]

class HeapNode():
	def __init__(self, curr_city, g, widgets,parent, use_dis=False):
		self.city = curr_city
		self.g = g
		self.h = 0
		self.widgets = widgets
		self.use_dis = use_dis
		self.parent = parent

	def get_hash(self):
		res = self.city
		for widget in self.widgets:
			res += '#' + ''.join(widget.left)
		return hash(res)

	def __cmp__(self, other):
		return self.get_fn() - other.get_fn()

	def get_fn(self):
		return self.g + self.h

	def isGoal(self):
		return sum([widget.get_left() for widget in self.widgets]) == 0

	def check_new(self, city):
		return any([widget.check_new(city) for widget in self.widgets])

	def update_h(self):
		if not self.use_dis:
			self.g += 1
		else:
			self.g += self.get_miles(self.parent.city, self.city)
		self.h = max([widget.get_left() for widget in self.widgets])

	def update_h2(self):
		max_left_miles = -1
		for widget in self.widgets:
			left_miles = 0
			for i, c in enumerate(widget.left):
				if i == 0:
					if self.city != '':
						left_miles += mile_matrix[ord(self.city) - ord('A')][ord(c) - ord('A')]
				else:
					left_miles += mile_matrix[ord(widget.left[i - 1]) - ord('A')][ord(c) - ord('A')]
			max_left_miles = max(max_left_miles, left_miles)
		self.h = max_left_miles


	def update_city(self, city):
		self.city = city
		for widget in self.widgets:
			widget.add_city(city)
		self.update_h()
		
	def get_miles(self, start, end):
		if start == '':
			return 0
		return mile_matrix[ord(start) - ord('A')][ord(end) - ord('A')]
