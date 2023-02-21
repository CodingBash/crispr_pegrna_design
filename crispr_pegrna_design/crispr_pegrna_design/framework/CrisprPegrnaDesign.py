import dataclasses

from pydantic import BaseModel, validate_arguments
from Bio.Seq import Seq
import re
from typeguard import typechecked
from typing import Union, List, Tuple, Mapping
import numpy as np
import random

# Python class of a Codon
@dataclasses.dataclass
class Codon:
    sequence_codon_index:int
    genome_index_start: int
    genome_index_stop: int
    codon_sequence: Union[Seq, str]
    
class Config:
    arbitrary_types_allowed = True

# Python function to split an ORF sequence into sets of codons
@validate_arguments(config=Config)
def split_into_codons(sequence: Union[Seq, str], codon_indices=None) -> List[Codon]:
    codon_set = []
    for codon_index, genomic_index in enumerate(range(int(len(sequence)/3))):
        codon = sequence[genomic_index*3:genomic_index*3+3]
        codon_set.append(Codon(sequence_codon_index=genomic_index, genome_index_start=genomic_index*3, genome_index_stop=genomic_index*3+3, codon_sequence=codon))
    return codon_set

# Class for containing codon set
class CodingTilingSequence(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        
    complete_sequence: Union[Seq, str]
    coding_coordinates: Tuple[int,int]
    tiling_coordinates: Tuple[int,int]
        
    
    coding_codon_set: List[Codon]
    tiled_codon_set: List[Codon]
    orf_position: int


# Define dictionary mapping codons to amino acid letter
dna_codon_map: Mapping[str, str] = {
    # 'M' - START, '_' - STOP
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TGT": "C", "TGC": "C",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TTT": "F", "TTC": "F",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "CAT": "H", "CAC": "H",
    "ATA": "I", "ATT": "I", "ATC": "I",
    "AAA": "K", "AAG": "K",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATG": "M",
    "AAT": "N", "AAC": "N",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TGG": "W",
    "TAT": "Y", "TAC": "Y",
    "TAA": "_", "TAG": "_", "TGA": "_"
}

# Codon frequencies based on Human table from GenScript https://www.genscript.com/tools/codon-frequency-table
# TODO: Double check that all frequencies and codons are accurate
# Originally to be used to select codon that is most frequent, however since all codons for a letter will be included in initial pegRNA set, this dictionary is not used
dna_codon_frequency_map: Mapping[str, List[Tuple[str, float]]]= {
    "F": [("TTT", 0.45), ("TTC", 0.55)],
    "L": [("TTA", 0.07), ("TTG", 0.13), ("CTT", 0.13), ("CTC", 0.20), ("CTA", 0.07), ("CTG", 0.41)],
    "Y": [("TAT", 0.43), ("TAC", 0.57)],
    "_": [("TAA", 0.28), ("TAG", 0.20,), ("TGA", 0.52)],
    "H": [("CAT", 0.41), ("CAC", 0.59)],
    "Q": [("CAA", 0.25), ("CAG", 0.75)],
    "I": [("ATT", 0.36), ("ATC", 0.48), ("ATA", 0.16)],
    "M": [("ATG", 1.00)],
    "N": [("AAT", 0.46), ("AAC", 0.54)],
    "K": [("AAA", 0.42), ("AAG", 0.58)],
    "V": [("GTT", 0.18), ("GTC", 0.24), ("GTA", 0.11), ("GTG", 0.47)],
    "D": [("GAT", 0.46), ("GAC", 0.54)],
    "E": [("GAA", 0.42), ("GAG", 0.58)],
    "S": [("TCT", 0.18), ("TCC", 0.22), ("TCA", 0.15), ("TCG", 0.06), ("AGT", 0.15), ("AGC", 0.24)],
    "C": [("TGT", 0.45), ("TGC", 0.55)],
    "W": [("TGG", 1.00)],
    "P": [("CCT", 0.28), ("CCC", 0.33), ("CCA", 0.27), ("CCG", 0.11)],
    "R": [("CGT", 0.08), ("CGC", 0.19), ("CGA", 0.11), ("CGG", 0.21), ("AGA", 0.20), ("AGG", 0.20)],
    "T": [("ACT", 0.24), ("ACC", 0.36), ("ACA", 0.28), ("ACG", 0.12)],
    "A": [("GCT", 0.26), ("GCC", 0.40), ("GCA", 0.23), ("GCG", 0.11)],
    "G": [("GGT", 0.16), ("GGC", 0.34), ("GGA", 0.25), ("GGG", 0.25)]
}
    
assert len(dna_codon_frequency_map.keys()) == 21 # Ensure all amino acids represented
codons = [codon[0] for codons in dna_codon_frequency_map.values() for codon in codons]
assert len(set(codons)) == len(codons) # Ensure no duplicate codons across amino acids
assert len(codons) == 4**3 # Ensure 64 codons 

# Ensure frequency sums to about 1
for codons in dna_codon_frequency_map.values():
    codons_sum = sum([codon[1] for codon in codons]) 
    assert codons_sum >= 0.99 and codons_sum <= 1.01


class HelperFunctions:
    '''Package helper functions'''

    @typechecked
    @staticmethod
    def calculate_hamming(sequence_A: Union[Seq,str], sequence_B: Union[Seq,str]) -> int:
        '''Calculate hamming distance between two sequences - naive approach'''
        difference = 0
        for i,nt_A in enumerate(sequence_A):
            if sequence_A[i].upper() != sequence_B[i].upper():
                difference = difference + 1
        return difference

    @typechecked
    @staticmethod
    def get_first_position_difference(sequence_A: Union[Seq,str], sequence_B: Union[Seq,str]) -> int:
        '''From two sequences, get position of first difference'''
        for i,nt_A in enumerate(sequence_A):
            if sequence_A[i].upper() != sequence_B[i].upper():
                return i
        return len(sequence_A)

    @typechecked
    @staticmethod
    def get_last_position_difference(sequence_A: Union[Seq,str], sequence_B: Union[Seq,str]) -> int:
        '''From two sequences, get position of last difference'''
        position = 0
        for i,nt_A in enumerate(sequence_A):
            if sequence_A[i].upper() != sequence_B[i].upper():
                position = i 
        return position

    @typechecked
    @staticmethod
    def get_all_position_differences(sequence_A: Union[Seq,str], sequence_B: Union[Seq,str]) -> List[int]:
        '''From two sequences, get all positions that are different'''
        positions = []
        for i,nt_A in enumerate(sequence_A):
            if sequence_A[i].upper() != sequence_B[i].upper():
                positions.append(i)
        return positions

    @typechecked
    @staticmethod
    def findall(substr:Union[Seq,str], mainstr:Union[Seq,str]):
        '''Find all positions of substr string in mainstr string'''
        return np.asarray([m.start() for m in re.finditer('(?={})'.format(str(substr).upper()), str(mainstr).upper())])

import numpy as np
import random

@dataclasses.dataclass
class MutatedCodon:
    original_codon: Codon
    mutated_codon: Codon
    hamming_distance: int

# Note 9/21/2022 - only just supporting one codon sample since for our case it is just the stop codon, could change to list of tuples to support more in the future
@typechecked
def mutate_sequence_primeeditingscreen(coding_tiling_sequence: CodingTilingSequence, dna_codon_frequency_map: Mapping[str, List[Tuple[str, float]]], dna_codon_map: Mapping[str, str], ignore_codon_letters: List[str] = ["_"], add_synonymous_mutations:bool=False, highest_frequent_codon:bool=False):
    '''
        Helper function for getting the final mutant sequence
    '''
    def get_mutant_sequence(codon_set_arg: List[Codon], codon_index = None) -> Tuple[str, str]:
        new_sequence = "".join([str(codon.codon_sequence) for codon in codon_set_arg])
        new_sequence_spaced = new_sequence
        if codon_index != None:
            new_sequence_left = "".join([str(codon.codon_sequence) for codon in codon_set_arg[:codon_index]])
            new_sequence_codon = str(codon_set_arg[codon_index].codon_sequence)
            new_sequence_right = "".join([str(codon.codon_sequence) for codon in codon_set_arg[codon_index+1:]])
            new_sequence_spaced = new_sequence_left +" "+new_sequence_codon + " " + new_sequence_right
        return new_sequence, new_sequence_spaced
            
    
    # This list will contain the set of all mutated sequences
    mutated_sequences: List[MutatedCodon] = []
    
    '''
        Add the WT oligo
    '''
    # Append the WT seqeunce
    wt_sequences_tuple = get_mutant_sequence(coding_tiling_sequence.tiled_codon_set)
    mutated_sequences.append((None, (None, None), (None, None), coding_tiling_sequence.tiled_codon_set, wt_sequences_tuple[0], wt_sequences_tuple[1], None))
    
    
    '''
        Iterate through each codon (which will be mutated to every other codon)
    '''
    tiled_codon_mutations:List[List[List[Mappable]]] = []
    tiled_codon: Codon
    codon_index: int
    for codon_index, tiled_codon in enumerate(coding_tiling_sequence.tiled_codon_set):
        print("Getting pegRNAs for codon {}/{}: {}".format(codon_index, len(coding_tiling_sequence.tiled_codon_set), tiled_codon))
        mutated_codon_list = list(dna_codon_frequency_map.keys()) 
        codon_letter = dna_codon_map[tiled_codon.codon_sequence]
        
        '''
            If not adding any synonymous mutations, then remove the same codon from the list to mutate to
        '''
        if not add_synonymous_mutations:
            mutated_codon_list.remove(codon_letter)
       
        '''
            Remove codons to ignore
        '''
        for ignored_codon_letter in ignore_codon_letters:
            mutated_codon_list.remove(ignored_codon_letter)
        
        '''
            Iterate through the list of codon variants "mutated_codon_list" to mutate to 
        '''
        mutated_codon_letter_mutations:List = []
        mutated_codon_letter: str
        for mutated_codon_letter in mutated_codon_list:
            print("\tAA Mutation {}>{}".format(codon_letter, mutated_codon_letter))
            '''
                Get all possible trinucleotide codon based on the residue letter
            '''
            possible_DNA_codons: List[Tuple[str, float]] = dna_codon_frequency_map[mutated_codon_letter]
            possible_DNA_codons = [possible_codon for possible_codon in possible_DNA_codons if possible_codon[0] != str(tiled_codon.codon_sequence)] # If synonymous codon letter, remove the exact same codon trinucleotide
            
            # If there are no codons to mutate to (for instance, the synonymous codon only has one codon trinucleotide which was removed)
            if len(possible_DNA_codons) == 0: 
                continue
              
            '''
                Determine the final list of trinucleotide codons to mutate for corresponding residue:
                i.e. only the highest frequent trinucleotide in the human genome or all the trinucleotides: 
            '''
            dna_codon_frequencies: List[float] = [possible_codon[1] for possible_codon in possible_DNA_codons]
            if highest_frequent_codon == True:
                mutated_DNA_codons:List[Seq] = [Seq(possible_DNA_codons[np.argmax(dna_codon_frequencies)][0])] # Get most frequent codon
            else:
                mutated_DNA_codons:List[Seq] = [Seq(codon[0]) for codon in possible_DNA_codons] # Get most frequent codon
        
            '''
                Create the MutatedCodon object list for each trinclueotide codon to mutate to. 
            '''
            mutated_codons:List[MutatedCodon] = []
            mutated_DNA_codon: Seq
            for mutated_DNA_codon in mutated_DNA_codons:
                mutated_codon: Codon = dataclasses.replace(tiled_codon)
                mutated_codon.codon_sequence = mutated_DNA_codon
                mutated_codon_obj = MutatedCodon(original_codon=tiled_codon, mutated_codon=mutated_codon, hamming_distance=calculate_hamming(mutated_codon.codon_sequence, tiled_codon.codon_sequence))
                mutated_codons.append(mutated_codon_obj)

                    
            '''
                We have generated all the mutated codons for saturation mutagenesis, now iterate through each one and create the 
                different PrimeDesign sequences that vary based on PAM-disrupting mutation and synonymous mutation
                
                This is where the main pegRNA design is done.
            '''
            mutated_codon_sequence_mutations: List[Mappable] = []
            mutated_codon_i: int
            mutated_codon: MutatedCodon
            for mutated_codon_i, mutated_codon in enumerate(mutated_codons):
                print("\t\tCodon Mutation {}>{}".format(str(mutated_codon.original_codon.codon_sequence), str(mutated_codon.mutated_codon.codon_sequence)))
                '''
                    Get the expected mutated sequence based on the mutated codon, along with its reverse complement
                    
                    This sequence is needed to geenerate the pegRNA, specifically determining where the nick site can be and if the mutant is already PAM-disrupting
                ''' 
                mutated_sequence = list(str(coding_tiling_sequence.complete_sequence))
                mutated_sequence[coding_tiling_sequence.tiling_coordinates[0]+mutated_codon.original_codon.genome_index_start:coding_tiling_sequence.tiling_coordinates[0]+mutated_codon.original_codon.genome_index_stop] = mutated_codon.mutated_codon.codon_sequence
                mutated_sequence = "".join(mutated_sequence)
                mutated_sequence_revcomp = str(Seq(mutated_sequence).reverse_complement())
                assert len(mutated_sequence) == len(coding_tiling_sequence.complete_sequence), "Ensure mutated and original sequence is same length, {}, {}".format(len(mutated_sequence), len(coding_tiling_sequence.complete_sequence))
                # Tunable parameter, max RTT considered. Set to 25 since literature suggests over 25 sacrifices efficiency
                max_rtt_length_considered=21
                
                '''
                     Get the position of the mutated truncleotide that is different (which determines the exact position the RT can start)
                '''
                first_position_difference = get_first_position_difference(mutated_codon.original_codon.codon_sequence, mutated_codon.mutated_codon.codon_sequence) # This is for the forward orientaion
                last_position_difference = get_last_position_difference(mutated_codon.original_codon.codon_sequence, mutated_codon.mutated_codon.codon_sequence) # This is for the reverse complement orientation
                assert first_position_difference >= 0 and first_position_difference < 3
                assert last_position_difference >= 0 and last_position_difference < 3
                
                complete_sequence = coding_tiling_sequence.complete_sequence
                assert str(mutated_codon.original_codon.codon_sequence) == complete_sequence[coding_tiling_sequence.tiling_coordinates[0] + mutated_codon.original_codon.genome_index_start:coding_tiling_sequence.tiling_coordinates[0] + mutated_codon.original_codon.genome_index_stop], "Ensure that codon is as expected based on coordinates, codon={}; coordinate_codon={}; coordinate_start={}, coordinate_end={}, codon={}".format(str(mutated_codon.original_codon.codon_sequence), complete_sequence[coding_tiling_sequence.tiling_coordinates[0] + mutated_codon.original_codon.genome_index_start:coding_tiling_sequence.tiling_coordinates[0] + mutated_codon.original_codon.genome_index_stop], coding_tiling_sequence.tiling_coordinates[0] + mutated_codon.original_codon.genome_index_start, coding_tiling_sequence.tiling_coordinates[0] + mutated_codon.original_codon.genome_index_stop, mutated_codon.original_codon)
                
                '''
                    Retrieve the possible coordinates of the RTT forward pegRNA
                '''
                RT_minimum_first_forward = coding_tiling_sequence.tiling_coordinates[0] + mutated_codon.original_codon.genome_index_start + first_position_difference
                RT_minimum_last_forward = coding_tiling_sequence.tiling_coordinates[0] + mutated_codon.original_codon.genome_index_start + last_position_difference
                most_upstream_nick_site_forward = RT_minimum_last_forward-max_rtt_length_considered
                most_downstream_nick_site_forward = RT_minimum_first_forward
                
                '''
                    Based on the possible RTT coordinates, get the possible PAM positions for the forward pegRNA
                '''
                possible_pam_sequence_forward_START = most_upstream_nick_site_forward+3 # This would place the last SNV to edit at the last position of the RTT, the +3 is pass the 3nt seed to the start PAM
                possible_pam_sequence_forward_END = most_downstream_nick_site_forward+6 # This would place the first SNV to edit at the first position of the RTT, the +6 is to pass the 3nt seed and 3nt PAM to end at the end of the PAM
                possible_pam_sequence_forward = str(complete_sequence[possible_pam_sequence_forward_START:possible_pam_sequence_forward_END])
                possible_pam_sequence_forward_coordinates = possible_pam_sequence_forward_START + findall("GG", possible_pam_sequence_forward[1:]) # 
                
                '''
                    Retrieve the possible coordinates of the RTT reverse pegRNA
                '''
                complete_sequence_revcomp = Seq(complete_sequence).reverse_complement()
                RT_minimum_first_reverse = len(complete_sequence_revcomp) - RT_minimum_first_forward -1 #TODO: Check for off by one error
                RT_minimum_last_reverse = len(complete_sequence_revcomp) - RT_minimum_last_forward -1 #TODO: Check for off by one error
                most_upstream_nick_site_reverse = RT_minimum_last_reverse-max_rtt_length_considered
                most_downstream_nick_site_reverse = RT_minimum_first_reverse
                
                '''
                    Based on the possible RTT coordinates, get the possible PAM positions for the reverse pegRNA
                '''
                possible_pam_sequence_reverse_START = most_upstream_nick_site_reverse+3 # This would place the last SNV to edit at the last position of the RTT
                possible_pam_sequence_reverse_END = most_downstream_nick_site_reverse+6 # This would place the first SNV to edit at the first position of the RTT
                possible_pam_sequence_reverse = str(complete_sequence_revcomp[possible_pam_sequence_reverse_START:possible_pam_sequence_reverse_END])
                possible_pam_sequence_reverse_coordinates = possible_pam_sequence_reverse_START + findall("GG", possible_pam_sequence_reverse[1:]) # 
                
                '''
                    STEP1 Check if the edit already mutates the PAM/seed by iterating through all possible PAMs.
                ''' 
                codon_start_position_forward = coding_tiling_sequence.tiling_coordinates[0] + mutated_codon.original_codon.genome_index_start
                codon_stop_position_forward = coding_tiling_sequence.tiling_coordinates[0] + mutated_codon.original_codon.genome_index_stop
                
                # TODO: There may be an off by one error here
                codon_stop_position_reverse = len(complete_sequence) - codon_start_position_forward # TODO: Check for off by one
                codon_start_position_reverse = len(complete_sequence) - codon_stop_position_forward # TODO: Check for off by one
                
                
                '''
                    STEP1A - iterate through forward PAMs to determine if disrupted by the edit
                '''
                is_edit_fwd_pam_disrupting = False
                is_edit_fwd_seed_disrupting = False
                pam_disrupted_forward_coordinates: List[int] = []
                for possible_pam_sequence_forward_coordinate in possible_pam_sequence_forward_coordinates: 
                    if codon_stop_position_forward >= possible_pam_sequence_forward_coordinate-3 and codon_start_position_forward <= possible_pam_sequence_forward_coordinate+3: # If PAM coord is within the codon coord
                        mutated_pam_sequence = mutated_sequence[possible_pam_sequence_forward_coordinate-3:possible_pam_sequence_forward_coordinate+3]
                        is_disrupted = False
                        is_current_edit_pam_disrupting = False
                        if mutated_pam_sequence[-2:] != "GG":
                            is_edit_fwd_pam_disrupting = True
                            is_current_edit_pam_disrupting = True
                            is_disrupted = True
                        if str(mutated_sequence[possible_pam_sequence_forward_coordinate-3:possible_pam_sequence_forward_coordinate]) != str(complete_sequence[possible_pam_sequence_forward_coordinate-3:possible_pam_sequence_forward_coordinate]):
                            is_edit_fwd_seed_disrupting = True
                            is_disrupted = True
                            
                        if is_disrupted:
                            pam_disrupted_forward_coordinates.append((possible_pam_sequence_forward_coordinate, is_current_edit_pam_disrupting)) # False for seed-disruption
                
                '''
                    STEP1B - iterate through reverse PAMs to determine if disrupted by the edit
                '''
                is_edit_rev_pam_disrupting = False
                is_edit_rev_seed_disrupting = False
                pam_disrupted_reverse_coordinates: List[int] = []
                for possible_pam_sequence_reverse_coordinate in possible_pam_sequence_reverse_coordinates: 
                    if codon_stop_position_reverse >= possible_pam_sequence_reverse_coordinate-3 and codon_start_position_reverse <= possible_pam_sequence_reverse_coordinate+3: # If PAM coord is within the codon coord
                        mutated_pam_sequence = mutated_sequence_revcomp[possible_pam_sequence_reverse_coordinate-3:possible_pam_sequence_reverse_coordinate+3]
                        is_disrupted = False
                        is_current_edit_pam_disrupting = False
                        if mutated_pam_sequence[-2:] != "GG":
                            is_edit_rev_pam_disrupting = True
                            is_current_edit_pam_disrupting = True
                            is_disrupted = True
                        if str(mutated_sequence_revcomp[possible_pam_sequence_reverse_coordinate-3:possible_pam_sequence_reverse_coordinate]) != str(complete_sequence_revcomp[possible_pam_sequence_reverse_coordinate-3:possible_pam_sequence_reverse_coordinate]):
                            is_edit_rev_seed_disrupting = True
                            is_disrupted = True
                            
                        if is_disrupted:
                            pam_disrupted_reverse_coordinates.append((possible_pam_sequence_reverse_coordinate, is_current_edit_pam_disrupting)) # False for seed-disruption
                
                forward_protospacers = []
                for pam_sequence_forward_coordinate, pam_disrupting in pam_disrupted_forward_coordinates:
                    print("\t\t\tPAM coordinate: {}".format(pam_sequence_forward_coordinate))
                    print("\t\t\tPAM: {}>{}".format(complete_sequence[pam_sequence_forward_coordinate-3:pam_sequence_forward_coordinate+3], mutated_sequence[pam_sequence_forward_coordinate-3:pam_sequence_forward_coordinate+3]))
                    forward_protospacers.append((complete_sequence[pam_sequence_forward_coordinate-20:pam_sequence_forward_coordinate+3], pam_sequence_forward_coordinate - codon_start_position_forward, pam_disrupting))
                reverse_protospacers = []
                for pam_sequence_reverse_coordinate, pam_disrupting in pam_disrupted_reverse_coordinates:
                    print("\t\t\tPAM coordinate: {}".format(pam_sequence_reverse_coordinate))
                    print("\t\t\tPAM: {}>{}".format(complete_sequence_revcomp[pam_sequence_reverse_coordinate-3:pam_sequence_reverse_coordinate+3], mutated_sequence_revcomp[pam_sequence_reverse_coordinate-3:pam_sequence_reverse_coordinate+3]))
                    reverse_protospacers.append((complete_sequence_revcomp[pam_sequence_reverse_coordinate-20:pam_sequence_reverse_coordinate+3], pam_sequence_reverse_coordinate-codon_start_position_reverse, pam_disrupting))
                
                '''
                    Add the mutations to the list
                '''
                mutated_codon_sequence_mutations.append({"tiled_mutated_codon": mutated_codon, "coding_mutated_codon": None, "is_edit_pam_disrupting": is_edit_fwd_pam_disrupting, "is_edit_seed_disrupting": is_edit_fwd_seed_disrupting, "orientation": "+", "protospacers": forward_protospacers})
                mutated_codon_sequence_mutations.append({"tiled_mutated_codon": mutated_codon, "coding_mutated_codon": None, "is_edit_pam_disrupting": is_edit_rev_pam_disrupting, "is_edit_seed_disrupting": is_edit_rev_seed_disrupting, "orientation": "-", "protospacers": reverse_protospacers})
                
                '''
                    If the desired edit is not PAM disrupting (which is typically the case that the edit is not PAM-disrupting), generate PAM-disrupting synonymous mutations
                '''
                generate_synonymous_pam_disruptions = True # To always generate synonymous mutations
                if (len(pam_disrupted_forward_coordinates) == 0) and (len(pam_disrupted_reverse_coordinates) == 0) or generate_synonymous_pam_disruptions:
                    print("\t\tMutated codon is not PAM-disrupting, now generating PAM-disrupting synonymous mutations")
                    # Contains list of PAM-disrupting synonymous mutations for fwd pegRNA: SynCodon, OrigCodon, pam_position
                    syn_codons_fwd_valid: List[Tuple[str, Codon, int]] = []
                    
                    '''
                        Iterate through all the PAM start coordinates that is within the PAM possible zone.
                    '''
                    for pam_sequence_forward_coordinate in possible_pam_sequence_forward_coordinates:
                        print("\t\t\tPAM forward coordinate: {}".format(pam_sequence_forward_coordinate))
                        '''
                            Ensure that the PAM is within the coding region, so that a synonymous mutation is able to be designed
                        '''
                        if (pam_sequence_forward_coordinate+3) > coding_tiling_sequence.coding_coordinates[0] and pam_sequence_forward_coordinate < coding_tiling_sequence.coding_coordinates[1]: 
                            '''
                                Get the codons that overlap the PAM, these are candidates for synonymous mutation for PAM disruption. These codons do not have to be within the tiling region.
                            '''
                            get_codon_start_absolute_fwd = lambda codon: coding_tiling_sequence.coding_coordinates[0] + codon.genome_index_start 
                            get_codon_stop_absolute_fwd = lambda codon: coding_tiling_sequence.coding_coordinates[0] + codon.genome_index_stop 
                            
                            # Get all codons codons that are within a PAM-disrupting sequencing
                            forward_codon_within_pegRNA_pam = [codon for codon in coding_tiling_sequence.coding_codon_set if (pam_sequence_forward_coordinate+1 >= get_codon_start_absolute_fwd(codon)) and (pam_sequence_forward_coordinate+1 < get_codon_stop_absolute_fwd(codon))] # NOTE 20221211 - I added a +1 since I want to check if the GG is within the codon, not the NGG
                            
                            '''
                                Iterate through each PAM-overlappng codons and get the possible synonymous codons - we will soon need to check that these synonymous mutations are actually PAM-disrupting
                            '''
                            for codon in forward_codon_within_pegRNA_pam:
                                print("\t\t\t\tOverlapping Codon: {},{}; PAM: {},{}".format(codon.codon_sequence, get_codon_start_absolute_fwd(codon), complete_sequence[pam_sequence_forward_coordinate-3:pam_sequence_forward_coordinate+5], pam_sequence_forward_coordinate))
                                codon_sequence = str(codon.codon_sequence)
                                
                                # This gets the synonymous codons
                                syn_codons: List[str] = [syn_codon for syn_codon,_ in dna_codon_frequency_map[dna_codon_map[codon_sequence]] if syn_codon != codon_sequence]
                                
                                '''
                                    Iterate through each PAM-overlapping codon for validation based on these criteria
                                    - Is it within the possible RTT
                                    - Does it actually disrupt the NGG PAM
                                '''
                                for syn_codon in syn_codons: 
                                    
                                    '''
                                        Get some positional information about the synonymous mutation
                                        - What is the first position of a difference between the original and mutated codon.
                                        - What is the last position of a difference between the original and mutated codon.
                                        - What are all positional differences between the original and mutated codon.
                                    '''
                                    syn_codon_first_position_difference = get_first_position_difference(codon_sequence, syn_codon) 
                                    syn_codon_last_position_difference = get_last_position_difference(codon_sequence, syn_codon)
                                    syn_codon_all_position_differences = get_all_position_differences(codon_sequence, syn_codon)
                                    
                                    syn_codon_first_position_difference_absolute = get_codon_start_absolute_fwd(codon) + syn_codon_first_position_difference
                                    syn_codon_last_position_difference_absolute =  get_codon_start_absolute_fwd(codon) + syn_codon_last_position_difference
                                    syn_codon_all_position_differences_absolute = [get_codon_start_absolute_fwd(codon) + position for position in syn_codon_all_position_differences]
                                    print("\t\t\t\tAbsolute: {}, PAM: {}".format(syn_codon_all_position_differences_absolute, pam_sequence_forward_coordinate))
                                    '''
                                        Ensure that the synonymous mutation is PAM disrupting
                                    '''
                                    # Just confirm that the expected region is a PAM
                                    assert coding_tiling_sequence.complete_sequence[pam_sequence_forward_coordinate+1:pam_sequence_forward_coordinate+3] == "GG" 
                                    
                                    mutates_pam = False
                                    for syn_codon_position_difference in syn_codon_all_position_differences_absolute:
                                        if (syn_codon_position_difference >= pam_sequence_forward_coordinate+1) and (syn_codon_position_difference <= pam_sequence_forward_coordinate+3):
                                            mutates_pam = True # TODO: Definitely need to validate the logic here with some print statements to ensure that the synonymous mutations are actually disrupting the PAM
                                    
                                    if mutates_pam != True:
                                        print("\t\t\t\t\tSynonymous mutation does not mutate PAM: {}>{}".format(codon.codon_sequence, syn_codon))
                                        continue # If the synonymous mutation does not mutate PAM, then skip and continue to checking the next synonymous mutation
                                    
                                    '''
                                        Ensure that the synonymous mutation is within the RTT
                                    '''
                                    within_RTT = (syn_codon_first_position_difference_absolute >= most_upstream_nick_site_forward) and (syn_codon_last_position_difference_absolute < most_downstream_nick_site_forward)  # TODO: Check for off-by-one error
                                    if within_RTT != True:
                                        print("\t\t\t\t\tSynonymous mutation not within RTT: {}>{}".format(codon.codon_sequence, syn_codon))
                                        continue # If the synonymous mutation is not within the RTT, then skip and continue to checking the next synonymous mutation
                                    
                                     # Get the target protospacer
                                    protospacer = [(complete_sequence[pam_sequence_forward_coordinate-20:pam_sequence_forward_coordinate+3], pam_sequence_forward_coordinate - codon_start_position_forward, True)]
                                
                                    print("\t\t\t\tPAM-disrupting synonymous mutation: {}>{}".format(codon.codon_sequence, syn_codon))
                                    #syn_codons_fwd_valid.append((syn_codon, codon, pam_sequence_forward_coordinate))
                                    
                                    mutated_syn_codon: Codon = dataclasses.replace(codon)
                                    mutated_syn_codon.codon_sequence = syn_codon
                                    mutated_syn_codon_obj = MutatedCodon(original_codon=codon, mutated_codon=mutated_syn_codon, hamming_distance=calculate_hamming(mutated_syn_codon.codon_sequence, codon.codon_sequence))
                                    mutated_codon_sequence_mutations.append({"tiled_mutated_codon": mutated_codon, "coding_mutated_codon": mutated_syn_codon_obj, "is_edit_pam_disrupting": is_edit_fwd_pam_disrupting, "is_edit_seed_disrupting": is_edit_fwd_seed_disrupting, "orientation": "+", "protospacers": protospacer})
                                
                    # Contains list of PAM-disrupting synonymous mutations for rev pegRNA: SynCodon, OrigCodon, pam_position
                    syn_codons_rev_valid: List[Tuple[str, Codon, int]] = []

                    '''
                        Iterate through all the PAM start coordinates that is within the PAM possible zone.
                    '''
                    for pam_sequence_reverse_coordinate in possible_pam_sequence_reverse_coordinates:
                        print("\t\t\tPAM reverse coordinate: {}".format(pam_sequence_reverse_coordinate))
                        '''
                            Ensure that the PAM is within the coding region, so that a synonymous mutation is able to be designed
                        '''
                        if (pam_sequence_reverse_coordinate+3) > (len(coding_tiling_sequence.complete_sequence) - coding_tiling_sequence.coding_coordinates[1]) and pam_sequence_reverse_coordinate < (len(coding_tiling_sequence.complete_sequence)-coding_tiling_sequence.coding_coordinates[0]):
                            '''
                                Get the codons that overlap the PAM, these are candidates for synonymous mutation for PAM disruption. These codons do not have to be within the tiling region.
                            '''
                            get_codon_start_absolute_rev = lambda codon: len(coding_tiling_sequence.complete_sequence) - (coding_tiling_sequence.coding_coordinates[0] + codon.genome_index_start)
                            get_codon_stop_absolute_rev = lambda codon: len(coding_tiling_sequence.complete_sequence) - (coding_tiling_sequence.coding_coordinates[0] + codon.genome_index_stop)


                            # Get all codons codons that are within a PAM-disrupting sequencing
                            reverse_codon_within_pegRNA_pam = [codon for codon in coding_tiling_sequence.coding_codon_set if (pam_sequence_reverse_coordinate+1 >= get_codon_stop_absolute_rev(codon)) and (pam_sequence_reverse_coordinate+1 < get_codon_start_absolute_rev(codon))] # NOTE 20221211 - I added a +1 since I want to check if the GG is within the codon, not the NGG

                            '''
                                Iterate through each PAM-overlappng codons and get the possible synonymous codons - we will soon need to check that these synonymous mutations are actually PAM-disrupting
                            '''
                            for codon in reverse_codon_within_pegRNA_pam:
                                print("\t\t\t\tOverlapping Codon: {},{}; PAM: {},{}".format(codon.codon_sequence.reverse_complement(), get_codon_stop_absolute_rev(codon), complete_sequence_revcomp[pam_sequence_reverse_coordinate-3:pam_sequence_reverse_coordinate+3], pam_sequence_reverse_coordinate))
                                codon_sequence = str(codon.codon_sequence)
                                codon_sequence_revcomp = str(codon.codon_sequence.reverse_complement())
                                # This gets the synonymous codons
                                syn_codons: List[str] = [syn_codon for syn_codon,_ in dna_codon_frequency_map[dna_codon_map[codon_sequence]] if syn_codon != codon_sequence]

                                '''
                                    Iterate through each PAM-overlapping codon for validation based on these criteria
                                    - Is it within the possible RTT
                                    - Does it actually disrupt the NGG PAM
                                '''
                                for syn_codon in syn_codons: 

                                    '''
                                        Get some positional information about the synonymous mutation
                                        - What is the first position of a difference between the original and mutated codon.
                                        - What is the last position of a difference between the original and mutated codon.
                                        - What are all positional differences between the original and mutated codon.
                                    '''
                                    syn_codon_revcomp = str(Seq(syn_codon).reverse_complement())
                                    syn_codon_first_position_difference = get_first_position_difference(codon_sequence_revcomp, syn_codon_revcomp) 
                                    syn_codon_last_position_difference = get_last_position_difference(codon_sequence_revcomp, syn_codon_revcomp)
                                    syn_codon_all_position_differences = get_all_position_differences(codon_sequence_revcomp, syn_codon_revcomp)

                                    syn_codon_first_position_difference_absolute = get_codon_stop_absolute_rev(codon) + syn_codon_first_position_difference
                                    syn_codon_last_position_difference_absolute =  get_codon_stop_absolute_rev(codon) + syn_codon_last_position_difference
                                    syn_codon_all_position_differences_absolute = [get_codon_stop_absolute_rev(codon) + position for position in syn_codon_all_position_differences]
                                    print("\t\t\t\tAbsolute: {}, PAM: {}".format(syn_codon_all_position_differences_absolute, pam_sequence_reverse_coordinate))
                                    '''
                                        Ensure that the synonymous mutation is PAM disrupting
                                    '''
                                    # Just confirm that the expected region is a PAM
                                    assert coding_tiling_sequence.complete_sequence.reverse_complement()[pam_sequence_reverse_coordinate+1:pam_sequence_reverse_coordinate+3] == "GG" 

                                    mutates_pam = False
                                    for syn_codon_position_difference in syn_codon_all_position_differences_absolute:
                                        if (syn_codon_position_difference >= pam_sequence_reverse_coordinate+1) and (syn_codon_position_difference <= pam_sequence_reverse_coordinate+3):
                                            mutates_pam = True # TODO: Definitely need to validate the logic here with some print statements to ensure that the synonymous mutations are actually disrupting the PAM

                                    if mutates_pam != True:
                                        print("\t\t\t\t\tSynonymous mutation does not mutate PAM: {}>{}".format(codon_sequence_revcomp, syn_codon_revcomp))
                                        continue # If the synonymous mutation does not mutate PAM, then skip and continue to checking the next synonymous mutation

                                    '''
                                        Ensure that the synonymous mutation is within the RTT
                                    '''
                                    within_RTT = (syn_codon_first_position_difference_absolute >= most_upstream_nick_site_reverse) and (syn_codon_last_position_difference_absolute < most_downstream_nick_site_reverse)  # TODO: Check for off-by-one error
                                    if within_RTT != True:
                                        print("\t\t\t\t\tSynonymous mutation not within RTT: {}>{}".format(codon_sequence_revcomp, syn_codon_revcomp))
                                        continue # If the synonymous mutation is not within the RTT, then skip and continue to checking the next synonymous mutation

                                    # Get the target protospacer
                                    protospacer = [(complete_sequence_revcomp[pam_sequence_reverse_coordinate-20:pam_sequence_reverse_coordinate+3], pam_sequence_reverse_coordinate - codon_start_position_reverse, True)]
                                    
                                    print("\t\t\t\tPAM-disrupting synonymous mutation: {}>{}".format(codon_sequence_revcomp, syn_codon_revcomp))
                                    #syn_codons_rev_valid.append((syn_codon, codon, pam_sequence_reverse_coordinate))
                                    
                                    mutated_syn_codon: Codon = dataclasses.replace(codon)
                                    mutated_syn_codon.codon_sequence = syn_codon
                                    mutated_syn_codon_obj = MutatedCodon(original_codon=codon, mutated_codon=mutated_syn_codon, hamming_distance=calculate_hamming(mutated_syn_codon.codon_sequence, codon.codon_sequence))
                                    mutated_codon_sequence_mutations.append({"tiled_mutated_codon": mutated_codon, "coding_mutated_codon": mutated_syn_codon_obj, "is_edit_pam_disrupting": is_edit_rev_pam_disrupting, "is_edit_seed_disrupting": is_edit_rev_seed_disrupting, "orientation": "-", "protospacers": protospacer})
            mutated_codon_letter_mutations.append((codon_letter, mutated_codon_letter, mutated_codon_sequence_mutations))
        tiled_codon_mutations.append((codon_index, tiled_codon, mutated_codon_letter_mutations))    
    return tiled_codon_mutations 