#!/usr/bin/perl
use utf8;
use strict;

# We want to be able to print unicode to the console
binmode(STDOUT, ":utf8");

# We'll read in forms from the corpus file
my $corpus_file = "English-FrischFeats.Segs.txt";
open (CORPUS, "<:utf8", $corpus_file) or die "Warning! Can't open corpus file: $!\n";


# Also, for readability, let's output the results in IPA
my $ipa_table = "disc2ipa.txt";
open (IPA, "<:utf8", $ipa_table) or die "Warning! Can't open file with IPA values: $!\n";

# A hash table to store IPA values
my %ipa;
while (my $line = <IPA>) {
	chomp($line);
	(my $disc, my $ipa) = split("\t",$line);
	$ipa{$disc} = $ipa;
}

# Now it's time to read the file and convert it

while (my $line = <CORPUS>) {
	# Remove final carriage return
	chomp($line);

	my @chars = split("", $line);
	for (my $c = 0; $c < scalar(@chars); $c++) {
		if ($ipa{$chars[$c]} ne '') {
			print $ipa{$chars[$c]};
		}
		else { print $chars[$c]; }
		print " "; ##### space delimited
	}
	print "\n";
}
close(CORPUS);
