# DNA is composed of four nucleotide bases: Adenine (A), Thymine (T), Cytosine (C), and Guanine (G)
nucleotides = ['A', 'T', 'C', 'G']

# A simple DNA sequence example
dna_sequence = 'ATGCGATACGTACG'

# Function to generate the complementary DNA strand
def generate_complementary_dna(sequence):
    complementary_dna = ''
    base_pair = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    for base in sequence:
        complementary_dna += base_pair[base]
    return complementary_dna

# Generate the complementary DNA strand
complementary_dna = generate_complementary_dna(dna_sequence)

# Transcribe DNA into RNA by replacing Thymine (T) with Uracil (U)
def transcribe_dna_to_rna(sequence):
    return sequence.replace('T', 'U')

# Transcribe the DNA sequence into RNA
rna_sequence = transcribe_dna_to_rna(dna_sequence)

# Define the genetic code for translation
genetic_code = {
    'AUG': 'M', 'UAA': '*', 'UAG': '*', 'UGA': '*',  # Start and stop codons
    'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',  # Phenylalanine, Leucine
    # ... (complete the genetic code for all 64 possible codons)
}

# Translate the RNA sequence into a protein sequence
def translate_rna_to_protein(sequence):
    protein_sequence = ''
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i+3]
        amino_acid = genetic_code[codon]
        protein_sequence += amino_acid
    return protein_sequence

# Translate the RNA sequence into a protein sequence
protein_sequence = translate_rna_to_protein(rna_sequence)

# Print the results
print(f'DNA Sequence: {dna_sequence}')
print(f'Complementary DNA: {complementary_dna}')
print(f'RNA Sequence: {rna_sequence}')
print(f'Protein Sequence: {protein_sequence}')
