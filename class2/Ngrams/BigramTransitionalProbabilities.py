# File to count bigram transitional probabilities in a file
import sys

celex_filename = "CelexLemmasInTranscription-DISC.txt"
celex = open(celex_filename, 'r')

output_filename = "CelexLemmasInTranscription-DISC.bigramtransprobs.txt"
output = open(output_filename, 'w')

# We'll make a list of all the lemmas
lemmas = []

# Now go through the celex file and get all the lemmas
for line in celex:
	lemma, freq, orthog, disc, phon2 = line.split("\t")
	lemmas.append(disc)

# Make a dictionary to store the counts
bigram_counts = {}
first_of_bigram_counts = {}

# Now go through the lemmas, counting all the bigrams
for lemma in lemmas:
	# Add word boundaries
	lemma = "#" + lemma + "#"
	# Step through the lemma
	for i in range(0, len(lemma)-1):
		# For each bigram, increase its count in the dictionary (the numerator)
		current_bigram = lemma[i:i+2]
		if current_bigram in bigram_counts:
			bigram_counts[current_bigram] += 1
		else:
			bigram_counts[current_bigram] = 1
			
		# Also, keep track of how often each segment occurs as the first member of a bigram (the denominator)
		if (lemma[i] in first_of_bigram_counts):
			first_of_bigram_counts[lemma[i]] += 1
		else:
			first_of_bigram_counts[lemma[i]] = 1

celex.close()
			
# Now get the counts and report them
# In order to sort in decreasing probability, it's useful to first calculate them all and then sort. We'll store them in a dictionary
transitional_bigram_probs = {}
# Go through the bigrams, calculate transitional probability and store in dictionary of transitional probabilities
for bigram, count in bigram_counts.items():
	bigram_trans_prob = float(count)/first_of_bigram_counts[bigram[0]]
	transitional_bigram_probs[bigram] = bigram_trans_prob
		
	
for bigram, bigram_trans_prob in sorted(transitional_bigram_probs.items(), key = lambda x: x[1], reverse=True):
	output.write( bigram + "\t{0:.8f}\n".format(bigram_trans_prob))
output.close()