# defines.py
# December 2016
#
# Paths, files, constants and tensor maps for Variant Filtration with Neural Nets.
# Everything that assumes the Broad file system should be kept to this file.
# Definitions shared across multiple files should be set here.
#
# Sam Friedman 
# sam@broadinstitute.org

data_path = '/dsde/data/deep/vqsr/'

data_dir = data_path + 'tensors/solexa_269365_ref_read_with_annotations/'
reference_fasta = data_path + 'Homo_sapiens_assembly19.fasta'

# VCF files
nist_vcf = data_path + 'vcfs/nist_na12878_minimal.vcf.gz'
negative_vcf = data_path + 'vcfs/recalibrated_g94982.vcf.gz'
hapmap_vcf = '/vcf/hapmap_3.3.b37.vcf.gz'
omni_vcf = data_path + 'vcfs/1000G_omni2.5.b37.vcf.gz'
exac_vcf = data_path + 'vcfs/exac_na12878.vcf.gz'
vqsr_vcf = data_path + 'vcfs/G94982.site_only_plus_mixins.vcf'
agilent_vcf = '/dsde/working/gauthier/ASHGposterResults/agilentParents_ICEoffspring_401trios_AS_VQSR.ICfiltered.recalibrated.vcf'
parent_vcf = '/dsde/working/gauthier/ASHGposterResults/ICEparents_agilentOffspring_401trios_AS_VQSR.ICfiltered.recalibrated.vcf'
gnomad_vcf = '/web/macarthurlab-distribution/gnomAD/release-170228/genomes/vcf/gnomad.genomes.r2.0.1.sites.1.vcf.gz'
mills_vcf = data_path + 'vcfs/Mills_and_1000G_gold_standard.indels.b37.vcf.gz'
dbsnp_vcf = '/dsde/working/sam/dbsnp_comparisons/dbsnp_150_hg19_All_20170403.vcf.gz'

# BED and BAM and other files
bam_file = data_path + 'bams/na12878_g94982_bamout_no_trim.bam'
gnomad_prefix = '/web/macarthurlab-distribution/gnomAD/release-170228/genomes/vcf/gnomad.genomes.r2.0.1.sites.'
nist_bed_file = data_path + 'beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed'
exon_bed_file = data_path + 'beds/gencode_exons_v24.bed'
repeat_bed_file = data_path + 'beds/repeat_masker_hg19.bed'
chrom_hmm_bed_file = data_path + 'beds/chrom_hmm.bed'
encode_gtf_file = data_path + 'gencode.v19.annotation.gtf_withproteinids'

# Total number of boolean bit-packed read flags, actual flags used is determined by the tensor map
# See https://broadinstitute.github.io/picard/explain-flags.html
read_flags = 12

mapping_quality_max = 60.0 # Mapping qualities from BWA are typically capped at 60

# Annotation sets
annotations = { 
				'_' : [], # Allow command line to unset annotations
				'gatk_w_qual' : ['MQ', 'DP', 'SOR', 'FS', 'QD', 'MQRankSum', 'QUAL', 'ReadPosRankSum'],
				'gatk' : ['MQ', 'DP', 'SOR', 'FS', 'QD', 'MQRankSum', 'ReadPosRankSum'],
				'best_practices' : ['MQ', 'DP', 'SOR', 'FS', 'QD', 'MQRankSum', 'ReadPosRankSum'],
				'annotations' : ['MQ', 'DP', 'SOR', 'FS', 'QD', 'MQRankSum', 'ReadPosRankSum'],
				'm2':['AF', 'AD_0', 'AD_1', 'MBQ', 'MFRL_0', 'MFRL_1', 'MMQ', 'MPOS'],
				'combine': ['MQ', 'DP', 'SOR', 'FS', 'QD', 'MQRankSum', 'ReadPosRankSum', 'AF', 'AD_0', 'AD_1', 'MBQ', 'MFRL_0', 'MFRL_1', 'MMQ', 'MPOS'],
				'gnomad': ['MQ', 'DP', 'SOR', 'FS', 'QD', 'MQRankSum', 'ReadPosRankSum', 'DP_MEDIAN', 'DREF_MEDIAN', 'GQ_MEDIAN', 'AB_MEDIAN'],
				}


# Base calling ambiguities, See https://www.bioinformatics.org/sms/iupac.html
ambiguity_codes = { 
					'K':[0,0,0.5,0.5], 'M':[0.5,0.5,0,0], 'R':[0.5,0,0,0.5], 'Y':[0,0.5,0.5,0], 'S':[0,0.5,0,0.5],
					'W':[0.5,0,0.5,0], 'B':[0,0.333,0.333,0.334], 'V':[0.333,0.333,0,0.334],'H':[0.333,0.333,0.334,0],
				  	'D':[0.333,0,0.333,0.334],'X':[0.25,0.25,0.25,0.25], 'N':[0.25,0.25,0.25,0.25],
				  	}

skip_char = '~'
indel_char = '*'

inputs = {'A':0, 'C':1, 'G':2, 'T':3}
inputs_indel = {'A':0, 'C':1, 'G':2, 'T':3, indel_char:4}
inputs_indel_both_cases = {'A':0, 'C':1, 'G':2, 'T':3, indel_char:4, 'a':0, 'c':1, 'g':2, 't':3}

# Defines 2-channel encoding of DNA 
# First channel is 1 for purine, 0 for pyrimidine.
# Second channel is pairing 1 for A,T, 0 for C,G.
dna_2bit = {'A':[1,1], 'C':[-1,-1], 'G':[1,-1], 'T':[-1,1], indel_char:2}


snp_labels = {'SNP':0, 'NOT_SNP':1}
indel_labels = {'INDEL':0, 'NOT_INDEL':1}
genotype_labels = {'HET', 'HOM_REF', 'HOM_ALT'}
snp_indel_labels = {'NOT_SNP':0, 'NOT_INDEL':1, 'SNP':2, 'INDEL':3}
calling_labels = {
					'REFERENCE':0, 'HET_SNP':1, 'HOM_SNP':2, 
					'HET_DELETION':3, 'HOM_DELETION':4, 
					'HET_INSERTION':5, 'HOM_INSERTION':6
				 }

base_labels_binary = { 'GOOD_BASE':0, 'BAD_BASE':1 }
bqsr_annotations = ['reverse', 'first_in_pair', 'mapping_quality', 'read_position']

reference_beds = [exon_bed_file, repeat_bed_file]
cigar_code = {'M':0, 'I':1, 'D':2, 'N':3, 'S':4}


architectures = {
				'2d':data_path+'tensors/g94982_allele_specific/', 
				'2d_annotations':data_path+'tensors/g94982_allele_specific/',
				'read_tensor':data_path+'tensors/g94982_allele_specific/',
				'1d':data_path+'tensors/g94982_na12878_1d_ws_128_ds/', 
				'bqsr':'/dsde/data/deep/bqsr/tensors/bqsr_bigger/', 
				'1d_annotations':data_path+'tensors/g94982_na12878_1d_ws_128_ds/',
				'mlp':data_path+'tensors/g94982_na12878_1d_ws_128_ds/',
				'1d_calling':data_path+'tensors/g94982_calling_tensors_1d/',
				'deep_variant':data_path+'tensors/g94982_deep_variant_ref_sort/',
				}


def annotations_from_args(args):
	if args.annotation_set and args.annotation_set in annotations:
		return annotations[args.annotation_set]
	return None



def get_tensor_channel_map_from_args(args):
	'''Return tensor mapping dict given args.tensor_map'''
	if not args.tensor_map:
		return None

	if 'read_tensor' == args.tensor_map:
		return get_tensor_channel_map_rt()
	elif '2d_2bit' == args.tensor_map:
		return get_tensor_channel_map_2bit()
	elif '1d_calling'== args.tensor_map:
		return get_tensor_channel_map_reference_reads()
	elif '2d' == args.tensor_map or '2d_annotations' == args.tensor_map or '2d_mapping_quality' == args.tensor_map:
		return get_tensor_channel_map_mq()
	elif 'reference' == args.tensor_map or '1d_dna' == args.tensor_map or '1d_annotations' == args.tensor_map:
		return get_tensor_channel_map_1d_dna()
	elif 'bqsr' == args.tensor_map:
		return bqsr_tensor_channel_map()
	elif 'annotations' == args.tensor_map:
		return annotations
	elif 'deep_variant' == args.tensor_map:
		return deep_variant_channel_map()
	else:
		raise ValueError('Unknown tensor mapping mode:', args.tensor_map)


def get_tensor_channel_map_1d_dna():
	'''1D Reference tensor with 4 channel DNA encoding.'''
	tensor_map = {}
	for k in inputs.keys():
		tensor_map[k] = inputs[k]
	
	return tensor_map


def get_tensor_channel_map_1d():
	'''1D Reference tensor with 4 channel DNA encoding'''
	tensor_map = {}
	for k in inputs.keys():
		tensor_map[k] = inputs[k]
	
	return tensor_map


def get_tensor_channel_map_1d_plus_beds():
	'''1D Reference tensor with 4 channel DNA encoding.
	Also includes channels for binary labels from bed files.
	'''
	tensor_map = {}
	for k in inputs.keys():
		tensor_map[k] = inputs[k]
	
	ref_offset = len(inputs)
	for i,b in enumerate(reference_beds):
		tensor_map[b] = ref_offset + i
	
	return tensor_map



def get_tensor_channel_map_reference_reads():
	'''Read and reference tensor with 4 channel DNA encoding.
	Plus insertions and deletions.
	'''
	tensor_map = {}
	for k in inputs_indel.keys():
		tensor_map['read_'+k] = inputs_indel[k]
	for k in inputs_indel.keys():
		tensor_map['reference_'+k] = len(inputs_indel) + inputs_indel[k]	
	
	return tensor_map

def get_tensor_channel_map():
	'''Read and reference tensor with 4 channel DNA encoding.
	Also includes read flags.
	'''
	tensor_map = {}
	for k in inputs_indel.keys():
		tensor_map['read_'+k] = inputs_indel[k]
	for k in inputs_indel.keys():
		tensor_map['reference_'+k] = len(inputs_indel) + inputs_indel[k]			
	tensor_map['flag_bit_4'] = 10
	tensor_map['flag_bit_5'] = 11	
	tensor_map['flag_bit_6'] = 12	
	tensor_map['flag_bit_7'] = 13
	return tensor_map


def get_tensor_channel_map_mq():
	'''Read and reference tensor with 4 channel DNA encoding.
	Also includes read flags.
	'''
	tensor_map = {}
	for k in inputs_indel.keys():
		tensor_map['read_'+k] = inputs_indel[k]
	for k in inputs_indel.keys():
		tensor_map['reference_'+k] = len(inputs_indel) + inputs_indel[k]			

	tensor_map['flag_bit_4'] = 10
	tensor_map['flag_bit_5'] = 11	
	tensor_map['flag_bit_6'] = 12	
	tensor_map['flag_bit_7'] = 13
	tensor_map['flag_bit_9'] = 14	
	tensor_map['flag_bit_10'] = 15

	tensor_map['mapping_quality'] = 16

	return tensor_map


def get_tensor_channel_map_rt():
	'''Read and reference tensor with 4 channel DNA encoding.
	Also includes read flags for strand and pair.
	'''
	tensor_map = {}
	for k in inputs_indel.keys():
		tensor_map['read_'+k] = inputs_indel[k]
	for k in inputs_indel.keys():
		tensor_map['reference_'+k] = len(inputs_indel) + inputs_indel[k]			

	tensor_map['flag_bit_4'] = 10
	tensor_map['flag_bit_5'] = 11	
	tensor_map['flag_bit_6'] = 12	
	tensor_map['flag_bit_7'] = 13

	tensor_map['mapping_quality'] = 14

	return tensor_map


def get_tensor_channel_map_2bit():
	'''Read and reference tensor with 2bit DNA encoding.
	Also includes read flags.
	'''	
	tensor_map = {}
	tensor_map['read_purine'] = 0
	tensor_map['read_pair'] = 1
	tensor_map['read_indel'] = 2
	tensor_map['reference_purine'] = 3
	tensor_map['reference_pair'] = 4
	tensor_map['reference_indel'] = 5	
	tensor_map['flag_bit_4'] = 6
	tensor_map['flag_bit_5'] = 7	
	tensor_map['flag_bit_6'] = 8	
	tensor_map['flag_bit_7'] = 9
	return tensor_map


def bqsr_tensor_channel_map():
	''' BQSR tensors are read and reference sequence.
	Each tensor includes args.window_size bases 
	preceding the base to predict.
	'''
	tensor_map = {}
	for k in inputs_indel.keys():
		tensor_map['read_'+k] = inputs_indel[k]
	for k in inputs_indel.keys():
		tensor_map['reference_'+k] = len(inputs_indel) + inputs_indel[k]			
	return tensor_map


def tensor_shape_from_args(args):
	in_channels = total_input_channels_from_args(args)
	if args.tensor_map == 'reference':
		tensor_shape = (args.window_size, in_channels)
	elif args.channels_last:
		tensor_shape = (args.read_limit, args.window_size, in_channels)
	else:
		tensor_shape = (in_channels, args.read_limit, args.window_size) 
	return tensor_shape


def deep_variant_channel_map():
	tensor_map = {}
	tensor_map['bases'] = 0
	tensor_map['reference'] = 1
	tensor_map['strand'] = 2
	return tensor_map


def total_input_channels_from_args(args):
	'''Get the number of channels in the tensor map'''		
	return len(get_tensor_channel_map_from_args(args))


def get_reference_and_read_channels(args):
	'''Get the number of read and reference channels in the tensor map'''		
	count = 0
	tm = get_tensor_channel_map_from_args(args)
	for k in tm.keys():
		if 'read' in k or 'reference' in k:
			count += 1
	return count
