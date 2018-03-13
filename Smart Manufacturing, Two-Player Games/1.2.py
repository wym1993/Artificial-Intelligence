import copy
from HeapNode import HeapNode
from widgetNode import WidgetNode
from component import *
import matplotlib.pyplot as plt
import sys

str_list = ["AEDCA", "BEACD", "BABCE", "DADBD", "BECBD"]
goal = [0, 0, 0, 0, 0]
cities = ['A', 'B', 'C', 'D', 'E']
target_mileage = 5473
mile_matrix = [[0, 1064, 673, 1401, 277], \
				[1064, 0, 958, 1934, 337], \
				[673, 958, 0, 1001, 399], \
				[1401, 1934, 1001, 0, 387], \
				[277, 337, 399, 387, 0]]

def check_goal(layer):
	for status in layer.status_list:
		if status.check_left() == goal:
			return True
	return False

def count_fluents(layer):
	curr_fluents = set()
	for status in layer.status_list:
		lefts = status.check_left()
		curr_fluents |= set([(i, lefts[i]) for i in range(len(lefts))])
	return curr_fluents

def get_mileage(path):
	total = 0
	for i in range(0, len(path) - 1):
		if path[i].city == '':
			continue
		total += mile_matrix[ord(path[i].city) - ord('A')][ord(path[i + 1].city) - ord('A')]
	return total

def get_path(layers):
	queue = []
	curr_idx = len(layers) - 1
	min_mileage = -1
	minpath, path = [], []

	for city in cities:
		last_status = layers[curr_idx].find_status_from_goal(goal, city)
		if last_status:
			queue.append((last_status, curr_idx, [last_status]))

	while queue:
		curr_status, curr_idx, path = queue.pop(0)
		tmp_mile = get_mileage(path)
		#print ''.join([status.city for status in path]), tmp_mile
		
		if curr_idx == 0:
			if min_mileage < 0 or tmp_mile < min_mileage:
				min_mileage = tmp_mile
				minpath = path[:]
		else:
			for status in curr_status.prev_status:
				queue.insert(0, (status, curr_idx - 1, [curr_status] + path))


	return minpath


def main():
	widgets = []
	for string in str_list:
		widgets.append(WidgetNode(string))

	# Add Init State
	layers = [Layer([Status('', widgets)])]
	has_reach_goal = False
	layer_nums, mile_nums = [], []

	while not check_goal(layers[-1]):
		'''
		print len(layers)
		if len(layers) in [2, 3]:
			print len(layers)
			for status in layers[-1].status_list:
				print status.city, [tmp.city for tmp in status.prev_status]
		if len(layers) > 3:
			return
		'''
		new_status = layers[-1].get_nxt_layer()
		layers.append(Layer(new_status))

	print 'minimum ' + str(len(layers) - 1) + ' levels required to found solution'
	has_reach = False

	while True:
		print 'At layer ' + str(len(layers) - 1)

		# Count the number of fluents at each layer
		curr_fluents = count_fluents(layers[-1])
		print 'Current number of fluents is ' + str(len(curr_fluents))

		# Get the shortest path
		path = get_path(layers)
		print 'Shortest path is ' + ''.join([status.city for status in path])[:-1]

		# Count total mileage
		total = get_mileage(path)
		print 'Total mileage is ' + str(total)

		# Store the mileage data for plot
		layer_nums.append(len(layers) - 1)
		mile_nums.append(total)
		print ''
		
		if total <= target_mileage:
			if has_reach:
				break;
			has_reach = True

		new_status = layers[-1].get_nxt_layer()
		layers.append(Layer(new_status))

	print 'search finish'
	plt.plot(layer_nums, mile_nums)
	plt.xlabel('Number of level')
	plt.ylabel('Total mileage')
	plt.show()

if __name__ == '__main__':
	main()
	
