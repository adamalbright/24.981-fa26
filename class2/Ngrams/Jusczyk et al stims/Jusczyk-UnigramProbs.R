# Read in the file.  The lines may be of different lengths, due to different numbers of segments. So, we need to tell R how many columns to expect. Here, we've specified up to 6 segments (perhaps not actually used), but we could add more if need be.
jusczyk1994 =  read.table("Jusczyk-UnigramProbs.txt", sep="\t", col.names = c("condition","item","logprob","seg1prob","seg2prob","seg3prob","seg4prob","seg5prob","seg6prob"),fill=TRUE)

aggregate(logprob ~ condition, data = jusczyk1994, mean)

library(beanplot)
dev.new(width=6,height=4)
par(mar=c(2,2,1,1))
beanplot(logprob~condition,data=jusczyk1994)
