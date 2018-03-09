class WidgetNode(object):
	def __init__(self, string):
		self.string = string
		self.used = []
		self.left = list(string)

	def get_left(self):
		return len(self.left)

	def add_city(self, c):
		if not self.left or self.left[0] != c:
			return

		self.used.append(self.left.pop(0))
		return
		
	def check_new(self, c):
		if not self.left or self.left[0] != c:
			return False
		return True

	def __str__(self):
		return "Widget " + self.string + ' now with left ' + ''.join(self.left)