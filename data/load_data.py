import csv
import sys
from collections import defaultdict

class LoadData:
	def __init__(self):
		self.words = {}
		self.target = {}
	
	def loadData(self):
		"""Retrieve the words from the file and add them into two dictionaries. 
		   The first one contains the mapping id:words while the second contains
		   the mapping id:target"""
		with open('/Users/lauranechita/mlpractical/data/train.csv') as csvfile:
			readCSV = csv.reader(csvfile, delimiter=',')
			for row in readCSV:				
					self.words[row[0]] = row[1];
					self.target[row[0]] = row[2];
			        
	