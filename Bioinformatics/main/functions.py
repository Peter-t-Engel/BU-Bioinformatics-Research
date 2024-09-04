import gzip
from Bio import SeqIO
from Bio.Seq import Seq
from Bio import motifs
import numpy as np
import pandas as pd
import logomaker as lm
import matplotlib.pyplot as plt

def getExonData(annotationFile):
    transcriptExons = {}
    with gzip.open(annotationFile, 'rt') as file: # "read text"
        for line in file:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if fields[2] == 'exon':
                chrom = fields[0]
                start = int(fields[3]) - 1 # fix indexing from 1-based to 0-based
                end = int(fields[4]) # end is inclusive in GTF so automatically adjusted
                strand = fields[6]
                attributes = fields[8] # get all the long semi-coloned information

                # some of the attributes have a section with something that looks like this: transcript_id "ENST00000424215";
                # so i need to check for "transcript_id" and then store the ENST000... ID thing in the transcriptID variable

                transcriptID = ""
                for attribute in attributes.split(';'):
                    if 'transcript_id' in attribute:
                        transcriptID = attribute.split('"')[1] # split the 'transcript_id "ENST000..."' and get the second element aka the transcript id
                        break

                if transcriptID not in transcriptExons:
                    transcriptExons[transcriptID] = [] # if its a new transcriptID initialize a list for exon coordinates under that transcript
        
                transcriptExons[transcriptID].append((chrom, start, end, strand))

    for id in list(transcriptExons.keys()):
        if len(transcriptExons[id]) <= 1: # check if there is more than one exon in transcript (single exon transcript doesn't use any splice sites)
            del transcriptExons[id] # remove the key and value for any transcript with a single exon

    return transcriptExons

def openGenomeFile(genomeFile): 
    genome = {}
    file = gzip.open(genomeFile, 'rt')
    for record in SeqIO.parse(file, "fasta"): # parse file and iterate through each record
        genome[record.id] = record.seq # store sequence in dictionary under its identifier
    return genome

def getSpliceSites(transcriptExons, genome):
    fivePrimeSites = []
    threePrimeSites = []
    for id, exons in transcriptExons.items():
        for indx, exon in enumerate(exons):
            chrom, start, end, strand = exon[0], exon[1], exon[2], exon[3]
            
            # Exclude every first exons 3' site and every last exons 5' site
            if strand == '+':
                if start >= 20 and indx != 0:  # make sure its not the first exon of transcript and that there is enough upstream nucleotides
                    threePrimeSites.append((chrom, start - 20, start + 3, strand))
                    
                if end < len(genome[chrom]) - 6 and indx != len(exons) - 1: # make sure its not the last exon of transcript
                    fivePrimeSites.append((chrom, end - 3, end + 6, strand))

            elif strand == '-':
                if end < len(genome[chrom]) - 20 and indx != 0:  # make sure its not the first exon of transcript and that there is enough downstream nucleotides
                    threePrimeSites.append((chrom, end - 3, end + 20, strand))

                if indx != len(exons) - 1 and start >= 6: # make sure its not the last exon of transcript
                    fivePrimeSites.append((chrom, start - 6, start + 3, strand))

    return set(fivePrimeSites), set(threePrimeSites) # remove any duplicates

def extractSequences(spliceSites, genome):
    sequences = []
    for chrom, start, end, strand in spliceSites:
        seq = genome[chrom][start:end]
        if strand == '-':
            seq = str(seq.reverse_complement()) # reverse compliment so 3 and 5 prime are oriented correctly for compliment sequence
        else:
            seq = str(seq)
        sequences.append(seq)
    return sequences

def makePWM(sequences):
    instances = []
    for seq in sequences:
        instances.append(Seq(seq)) # make everything into Seq objects for biopython motifs
    motif = motifs.create(instances)
    PWM = motif.counts.normalize() # change nucleotides into # of each and then into probabilities of each
    return PWM

def makeDataFrame(PWM):
    pwmList = []
    for base in "ACGT":
        pwmList.append(PWM[base]) # append row of probabilities for each base
    pwmArray = np.array(pwmList).T # make list into a np array and with probabilities for each base going in columns (transposed)
    dataFrame = pd.DataFrame(pwmArray, columns = list("ACGT")) # create a DataFrame from the np array with columns labeled A,C,G,T
    return dataFrame

def plotLogo(dataFrame, title):
    logo = lm.Logo(dataFrame)
    logo.ax.set_title(title)
    plt.savefig(f"./plots/{title} logo final")

def createPSSM(sequences):
    instances = []
    for seq in sequences:
        instances.append(Seq(seq)) # make everything into Seq objects for biopython motifs
    motif = motifs.create(instances)

    # Calculate total counts at each position
    totalCounts = np.zeros(motif.length)
    for i in range(motif.length):
        positionSum = 0
        for nt in 'ACGT': # at each position in the motif go through each nucleotide and add up all counts at said position
            positionSum += motif.counts[nt][i] 
        totalCounts[i] = positionSum # append the summed count at position "i" in the motif to the list of total counts

    # calculate log odds scores for PSSM
    PSSM = np.zeros((motif.length, 4))
    for i, nt in enumerate('ACGT'): # "i" will go from 0-3 representing columns with each nucleotide "ACGT"
        for j in range(motif.length): # "j" will go from 0-8 or 0-22 depending on 5' or 3' and represents the row or position in the sequence
            count = motif.counts[nt][j] # get the count of nucleotide "nt" at position "j"
            PSSM[j, i] = np.log2((count / totalCounts[j]) / 0.25) # at coordinate (j, i) in the matrix add the log odds score of that nucleotide at that position (formula for log odds found online)

    return PSSM

def getSpliceSiteStrength(seq, PSSM):
    strength = 0
    for pos, base in enumerate(seq):
        if base == 'A':
            strength += PSSM[pos][0]
        elif base == 'C':
            strength += PSSM[pos][1]
        elif base == 'G':
            strength += PSSM[pos][2]
        elif base == 'T':
            strength += PSSM[pos][3]
    return strength



