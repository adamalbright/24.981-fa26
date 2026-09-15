from transliterate import transliterate
def apply_rules(inputs, rules, geminates_long):

	outputs = []

	for i in range(0,len(inputs)):
		word = transliterate(inputs[i], rules, geminates_long)		
		outputs.extend(word)
		
	return outputs