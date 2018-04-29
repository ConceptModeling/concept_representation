
import re

""" Returns a dictionary of {keyword: neighborhood word distribution}
M is the range of the word neighborhood: M words before the keyword + M words after.
"""
def neighborhood(input_file, M):
	splitOn = " "
	with open(input_file, "r") as lines:
		array = []
		for l in lines:
			tokens = l.split('\t')
			if tokens[0] not in ".,!@#$%^&*()[]{};:/?~`=+<>":
				array.append((tokens[0], tokens[1].strip('\n')))
	
	#print(array)
	nhoods = {}
	keyword = ""
	n = ""
	i=0
	while i < len(array):
		pair = array[i]
		count = i
		while pair[1] == "B" or pair[1]== "I":
			keyword += pair[0] + " "
			count += 1
			pair = array[count]
		if len(keyword) > 0:
			#print(keyword)
			n = " ".join(str(x) for x in [" ".join(p[0] for p in array[max(0,i-M):min(count+M, len(array))])])
			#n = [p[0] for p in array[max(0,i-M):min(count+M, len(array))]]
			#print("NEW: ", len(n), count-i)
			nhoods[keyword] = n

		
		keyword = ""
		n = ""
		i = count+1

	print(nhoods)
	return nhoods


neighborhood("./tagged_txt/A Brief Introduction to Neural Networks.txt", 10)

""" Returns a list of sets of keyword and phrases
Each set in the list describes one concept. Use KL-divergence with threshold T to 
make decision whether to group two words or not.
"""
def clusterKeywords(input_neighborhoods, T):
	
	return