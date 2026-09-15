# File to count probabilities of bigrams in a file
import sys

celex_filename = "CelexLemmasInTranscription-DISC.txt"
celex = open(celex_filename, 'r')

output_filename = "CelexLemmasInTranscription-DISC.bigramprobs.txt"
output = open(output_filename, 'w')

# We'll make a list of all the lemmas
lemmas = []

# Now go through the celex file and get all the lemmas
for line in celex:
	lemma, freq, orthog, disc, phon2 = line.split("\t")
	lemmas.append(disc)

# Make a dictionary to store the counts
bigram_counts = {}

# Now go through the lemmas, counting all the bigrams
total_bigrams = 0
for lemma in lemmas:
	# Add word boundaries
	lemma = "#" + lemma + "#"
	# Step through the lemma
	for i in range(0, len(lemma)-1):
		# For each bigram, increase its count in the dictionary
		current_bigram = lemma[i:i+2]
		total_bigrams += 1
		if current_bigram in bigram_counts:
			bigram_counts[current_bigram] += 1
		else:
			bigram_counts[current_bigram] = 1

celex.close()
			
# Now get the counts and report them
for bigram, count in sorted(bigram_counts.items(), key = lambda x: x[1], reverse=True):
	bigram_prob = float(count)/total_bigrams
	output.write( bigram + "\t{0:.12f}\n".format(bigram_prob))
output.close()