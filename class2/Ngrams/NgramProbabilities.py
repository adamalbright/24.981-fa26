# File to count positional bigram probabilities of test words in a file
import sys
import math

celex_filename = "CelexLemmasInTranscription-DISC.txt"
celex = open(celex_filename, 'r')

unigrams_filename = "Celex.UnigramProbabilities.txt"
unigrams_file = open(unigrams_filename, 'w')

bigrams_filename = "Celex.BigramProbabilities.txt"
bigrams_file = open(bigrams_filename, 'w')

bigrams_transitional_filename = "Celex.TransitionalBigramProbabilities.txt"
bigrams_transitional_file = open(bigrams_transitional_filename, 'w')


bigrams_positional_filename = "Celex.PositionalBigramProbabilities.txt"
bigrams_positional_file = open(bigrams_positional_filename, 'w')


test_filenames = ["AlbrightHayes2003.DISC.txt"]
testoutput_filename = "AlbrightHayes2003.Predictions.txt"



####### First read in the corpus
# We'll make a list of all the lemmas
lemmas = []

# Now go through the celex file and get all the lemmas
for line in celex:
	lemma_id, freq, orthog, disc, phon2 = line.split("\t")
	lemmas.append(disc)
celex.close()

####### Now count the corpus. (There's no real reason to do this separately, we could have done it while reading in the corpus.)
# Make dictionaries to store the counts. We'll calculate unigram probability, average bigram probability, transitional bigram probability, and positional bigram probability
unigram_counts = {}
total_unigrams = 0;
bigram_counts = {}
first_of_bigram_counts = {}
total_bigrams = 0;
positional_bigram_counts = []
total_positional_bigrams = []


# Go through the lemmas, counting all the characters
for lemma in lemmas:
	lemma = "[" + lemma + "]"
	# Step through the lemma
	for i in range(0, len(lemma)-1):
		# For each character and bigram, increase their count in the relevant dictionaries
		current_unigram = lemma[i]
		current_bigram = lemma[i:i+2]

		# Store the overall (position-independent) unigram count
		if (current_unigram in unigram_counts):
			unigram_counts[current_unigram] += 1
		else:
			unigram_counts[current_unigram] = 1
		total_unigrams += 1
		
		# Store the overall (position-independent) bigram count
		if (current_bigram in bigram_counts):
			bigram_counts[current_bigram] += 1
		else:
			bigram_counts[current_bigram] = 1
		
		# And add one to the total tally of bigrams, so we can calculate probabilities
		total_bigrams += 1
			
		# Also, the count for first members of bigrams. (Very similar to unigram counts, but the last segment of the word is omitted)
		if (current_unigram in first_of_bigram_counts):
			first_of_bigram_counts[current_unigram] += 1
		else:
			first_of_bigram_counts[current_unigram] = 1
		
		
		# Store the positional counts. These are stored in a list of dictionaries, where each member is the count for the corresponding position in the word.  If we've never seen a word this long before, add enough space for the current position. In theory, we should always have seen at least position n-1 before (on a previous iteration of this loop), but just to be safe, add however many positions are necessary to make the list of positional dictionaries long enough to accommodate the current bigram position.
		while len(positional_bigram_counts) <= i:
			positional_bigram_counts.append( {} )
			total_positional_bigrams.append( 0 )
		# Now store the current bigram in the dictionary for this position
		if (current_bigram in positional_bigram_counts[i]):
			positional_bigram_counts[i][current_bigram] += 1
		else:
			positional_bigram_counts[i][current_bigram] = 1

		# Finally, keep track of how many bigrams we've seen in this position, so that we can calculate the positional probability
		total_positional_bigrams[i] += 1
		# End of scanning through bigrams of the word

	# There's one last unigram, that we didn't hit by scanning through the bigrams. Count that last one, too.  If we've added word boundaries, then this last one really doesn't matter much, but we might as well be thorough.
	current_unigram = lemma[len(lemma)-1]
	if (current_unigram in unigram_counts):
		unigram_counts[current_unigram] += 1
	else:
		unigram_counts[current_unigram] = 1
	total_unigrams += 1
	# End of scanning through this word. Now we go back to the start of the loop, continuing with the next word

			
######## Now get the probabilities and report them
# First the unigram probabilities
unigram_probs = {}
for char, count in sorted(unigram_counts.items(), key = lambda x: x[1], reverse=True):
	unigrams_file.write( char + "\t" + str(float(count)/total_unigrams) + "\n")
	# store the probability
	unigram_probs[char] = float(count)/total_unigrams
unigrams_file.close()

# Now the bigram probabilities
bigram_probs = {}
for bigram, count in sorted(bigram_counts.items(), key = lambda x: x[1], reverse=True):
	bigram_prob = float(count)/total_bigrams
	bigrams_file.write( bigram + "\t{0:.12f}\n".format(bigram_prob))
	# store the probability
	bigram_probs[bigram] = bigram_prob
bigrams_file.close()

# Now the transitional bigram probabilities
transitional_bigram_probs = {}
for bigram, count in sorted(bigram_counts.items(), key = lambda x: x[1], reverse=True):
	transitional_bigram_prob = float(count)/first_of_bigram_counts[bigram[0:1]]
	bigrams_transitional_file.write( bigram + "\t{0:.12f}\n".format(transitional_bigram_prob))
	# store the probability
	transitional_bigram_probs[bigram] = transitional_bigram_prob
bigrams_transitional_file.close()

positional_bigram_probs = []
for bigram, count in sorted(bigram_counts.items(), key = lambda x: x[1], reverse=True):
#positional_bigram_counts = []
#total_positional_bigrams = []
	bigrams_positional_file.write( bigram );
	for i in range(0, len(positional_bigram_counts)):
		# Calculate the positional bigram probability
		if bigram in positional_bigram_counts[i]:
			bigram_prob = float(positional_bigram_counts[i][bigram])/total_positional_bigrams[i]
		else:
			bigram_prob = float(0)
		
		# Now store the probability.  If the positional_bigram_probs list isn't long enough, extend it
		while len(positional_bigram_probs) <= i:
			positional_bigram_probs.append( {} )
			
		# Store the bigram probability.  Unlike when we're counting the corpus, now we encounter each bigram in each position only once, so we don't need to check whether we've seen it before
		positional_bigram_probs[i][bigram] = bigram_prob
			
		bigrams_positional_file.write( "\t{0:.8f}".format(bigram_prob))		
	bigrams_positional_file.write("\n")		
		
bigrams_positional_file.close()


######## Now go through the files of test words, to calculate average (positional) bigram probabilities
test_output = open(testoutput_filename, 'w')

# Print a header row
test_output.write("Condition\tItem\t1gram joint prob\t2gram trans prob\t2gram avg prob\t2gram avg pos prob\n")

for test_filename in test_filenames:
	test_file = open(test_filename, 'r')
	# Aesthetic: differentiate which test forms came from which file, but label them according to the filename minus the .txt suffix
	condition = test_filename.replace(".txt","")

	for line in test_file:
		line = line.strip()
		# We assume that each line is a single word
		
		# Now we'll calculate the following:
		# 1. Joint unigram probability
		# 2. Joint transitional bigram probability
		# 3. Average bigram probability
		# 4. Average positional bigram probability
		
		# Calculate the joint unigram and transitional bigram probability 
		unigram_prob = 1
		transitional_bigram_prob = 1

		# Keep a running tally of the bigram probabilities
		total_bigram_prob = 0
		# And a running tally of the positional bigram probabilities
		total_positional_bigram_prob = 0
		# And a list of the individual bigram probabilities, so we can print them later. We could do this positionally, or with transitional probability, but let's just do regular bigram probabilities for now.
		bigram_probs_list = []
		
		# We should really check whether the word ("line") is longer than the known positions, since we'll get an index out of range error when we try to look up bigrams for positions past what we saw in the corpus.  For the current application, the test words are very short though, so we'll ignore this possibility for now.
		for i in range(0, len(line)-1):
			unigram = line[i]
			if unigram in unigram_probs:
				unigram_prob *= unigram_probs[unigram]
			else:
				unigram_prob = 0
								
		# For bigram probabilities, add word boundaries
		line = "[" + line + "]"
		for i in range(0, len(line)-2):
			bigram = line[i:i+2]
			
			if bigram in transitional_bigram_probs:
				transitional_bigram_prob *= transitional_bigram_probs[bigram]
			else:
				transitional_bigram_prob = 0

			if bigram in bigram_probs:
				bigram_prob = bigram_probs[bigram]
				total_bigram_prob += bigram_prob
				bigram_probs_list.append( "{0:.12f}".format(bigram_prob))
			# Otherwise, nothing is added to the running tally


			if bigram in positional_bigram_probs[i]:
				total_positional_bigram_prob += positional_bigram_probs[i][bigram]
			# Otherwise, nothing is added to the running tally
				
					
		# We now have the total probabilities, and the individual bigram probabilities.
		# First, average the bigram and positional bigram probabilities
		avg_bigram_prob = total_bigram_prob / (len(line)-2)
		avg_positional_bigram_prob = total_positional_bigram_prob / (len(line)-2)
		
		# Now output a line to the file. Start with the condition and the test string
		test_output.write(condition + "\t" + line)
		
		# Now the joint unigram and transitional bigram probabilities
		test_output.write("\t{0:.12f}".format(unigram_prob))
		test_output.write("\t{0:.12f}".format( transitional_bigram_prob ))
		
		# Now the average bigram and average positional bigram probabilities
		test_output.write("\t{0:.12f}".format( avg_bigram_prob ))
		test_output.write("\t{0:.12f}".format( avg_positional_bigram_prob ))
		
		# Finally, the list of the bigram probabilities
		test_output.write("\t" + "\t".join(bigram_probs_list) + "\n")
