# Infrastructure for a script to learn a rule ordering that converts Italian orthography to a more `phonetic' representation. It takes a set of known rules (in random order), and tries to find a compatible ordering.
# This version does not have any search implemented
import re
import sys
# Two useful functions defined for this purpose
from transliterate import *
from count_correct import *


# An option to mark geminates with a colon
geminates_long = True

# A file with the input to convert
input_filename = "italian-words.txt"
# And a file with the rules to apply
rules_filename = "ItalianRules.txt"

# Also, for convenience in checking whether our rules are working right, a file with the answer
check_filename = "italian-words.phonetic.txt"

# First, we read in the rules
rules_file = open(rules_filename, 'r')

# A list to store the rules_file
rules = []
# Read in the rules file and store the rules to a list
for line in rules_file:
	line = line.strip("\n")
	new_rule = line.split("\t")
	# Add the new rule to the list of rules
	rules.append(new_rule)
rules_file.close()

# Now read in the input file
input_file = open(input_filename, 'r')
# We'll make a list of 'inputs' to convert
inputs = []
for line in input_file:
	line = line.strip().lower()
	words = line.split()
	# Add this list of elements to the inputs list
	inputs.extend(words)
input_file.close()

# Now read in the file of correct answers
check_file = open(check_filename, 'r')
# We'll make a list of 'answers', to compare against current predictions
answers = []
for line in check_file:
	line = line.strip().lower()
	words = line.split()
	# Add this list of elements to the inputs list
	answers.extend(words)
check_file.close()

# A rule order is "consistent" if it generates the same output for all of the inputs as the given answer. That is, we want want the number of correctly generated forms to equal the total number of forms

# First, a sanity check, to make sure no user error. The number of inputs and answers should match. It not, squawk and give up.
if len(inputs) != len(answers):
	print ("Warning! different numbers of inputs (%s) and outputs (%s). Cannot continue." % (len(inputs), len(answers)))
	sys.exit()

# Apply the rules
outputs = transliterate_list(inputs, rules, geminates_long)	
print('\n'.join(outputs))

# See how many are correct  (this is not terribly efficient, since we re-apply the rules to the list)
number_correct = count_correct(inputs, answers, rules, geminates_long)

print ("%s correct out of %s" % (number_correct, len(inputs)))