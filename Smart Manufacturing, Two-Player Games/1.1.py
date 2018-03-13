import heapq as hq
from copy import deepcopy
from HeapNode import *
from widgetNode import WidgetNode

str_list = ["AEDCA", "BEACD", "BABCE", "DADBD", "BECBD"]
cities = ['A', 'B', 'C', 'D', 'E']
mile_matrix = [[0, 1064, 673, 1401, 277], \
				[1064, 0, 958, 1934, 337], \
				[673, 958, 0, 1001, 399], \
				[1401, 1934, 1001, 0, 387], \
				[277, 337, 399, 387, 0]]

def calc_miles(seq):
	dis = 0
	for i, c in enumerate(seq):
		if i == 0:
			continue
		dis += mile_matrix[ord(seq[i - 1]) - ord('A')][ord(c) - ord('A')]
	return dis

def search(widgets, is_mile=False):
	g_dict = {}
	path = {}
	heap = []
	steps = 0
	last = None
	
	startnode = HeapNode("", 0, widgets, None, is_mile)
	startnode.h = 0
	start_hash = startnode.get_hash()
	g_dict[start_hash] = startnode.g
	path[start_hash] = None
	hq.heappush(heap, startnode)

	while heap:
		curr = hq.heappop(heap)
		steps += 1

		if curr.isGoal():
			last = curr
			break

		# Find children of current node
		for city in cities:

			nxt = deepcopy(curr)
			nxt.parent = curr

			if city == curr.city:
				continue
			
			nxt.update_city(city)
			nxt_hash = nxt.get_hash()

			if nxt_hash not in g_dict or nxt.g < g_dict[nxt_hash]:
				g_dict[nxt_hash] = nxt.g
				path[nxt_hash] = curr
				hq.heappush(heap, nxt)

	# Find the path
	rev_path_list = ''
	while last != startnode:
		rev_path_list += last.city
		last = path[last.get_hash()]
	
	if not is_mile:
		return rev_path_list[::-1], len(rev_path_list), steps
	else:
		return rev_path_list[::-1], calc_miles(rev_path_list[::-1]), steps

def main():
	widgets = []
	for string in str_list:
		widgets.append(WidgetNode(string))

	path, l, steps = search(widgets)
	print 'Question 1'
	print 'Total number of nodes expanded is ' + str(steps)
	print 'Number of stop is ' + str(l)
	print 'Path of truck is ' + path
	print ''
	path, m, steps = search(widgets, True)
	print 'Question 2'
	print 'Total number of nodes expanded is ' + str(steps)
	print 'Number of miles is ' + str(m)
	print 'Path of truck is ' + str(path)


if __name__ == '__main__':
	main()
	#print calc_miles("ABAEDCADBCDEBB")





