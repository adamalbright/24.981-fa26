# File to count frequencies of segments in a file
import sys

celex_filename = "CelexLemmasInTranscription-DISC.txt"
celex = open(celex_filename, 'rU')

# We'll make a list of all the lemmas
lemmas = []

# Now go through the celex file and get all the lemmas
for line in celex:
	lemma, freq, orthog, disc, phon2 = line.split("\t")
	lemmas.append(disc)

# Make a dictionary to store the counts
unigram_counts = {}

# Now go through the lemmas, counting all the characters
for lemma in lemmas:
	# Step through the lemma
	for i in xrange(0, len(lemma)):
		# For each character, increase its count in the dictionary
		current_char = lemma[i]
		if current_char in unigram_counts:
			unigram_counts[current_char] += 1
		else:
			unigram_counts[current_char] = 1
			
# Now get the counts and report them
#for char, count in unigram_counts.items():
for char, count in sorted(unigram_counts.items(), key = lambda x: x[1], reverse=True):
	print char + "\t" + str(count)