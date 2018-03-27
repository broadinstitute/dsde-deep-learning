#!/bin/bash
ECHO=

MODE=write_tensors

# Tensor definition
TENSOR_MAP=read_tensor
READ_LIMIT=128
WINDOW_SIZE=128
BASE_QUALITY_MODE=phot
CHANNEL_ORDER=channels_last
ANNOTATION_SET=best_practices

# Downsample certain types of variant
DOWNSAMPLE_SNPS=0.05
DOWNSAMPLE_INDELS=0.5
DOWNSAMPLE_NOT_SNPS=1.0
DOWNSAMPLE_NOT_INDELS=1.0
DOWNSAMPLE_HOMOZYGOUS=1.0

REFERENCE=/seq/references/Homo_sapiens_assembly19/v1/Homo_sapiens_assembly19.fasta

#SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/chrom1_subset.interval_list
#SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/chrom1_10m_split.interval_list
SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome.interval_list
#SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list


# Genomes in a Bottle Project NA24385 Ashkenazi son
#DOWNSAMPLE_SNPS=0.001
#DOWNSAMPLE_INDELS=0.01
# DATA_DIR=/dsde/data/deep/vqsr/tensors/big_mix_channels_last/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24385.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/hg002_na24385_nist_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/hg002_na24385_nist_150bp_50x_all_calls.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/HG002_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_CHROM1-22_v3.2.2_highconf.bed


# 10X NA24385 Ashkenazi son
# DOWNSAMPLE_SNPS=0.001
# DOWNSAMPLE_INDELS=0.01
# DOWNSAMPLE_NOT_SNPS=1.0
# DOWNSAMPLE_NOT_INDELS=0.1
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94882_na24385_10x/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24385.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/na24385_10x_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na24385_10x.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/HG002_GIAB_highconf_IllFB-IllGATKHC-CG-Ion-Solid_CHROM1-22_v3.2.2_highconf.bed


# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, replicate 1
# DOWNSAMPLE_SNPS=0.005
# DOWNSAMPLE_INDELS=0.02
# DOWNSAMPLE_NOT_SNPS=0.1
# DOWNSAMPLE_NOT_INDELS=0.05
# TRAIN_VCF=/dsde/data/hybrid.m37m.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/hybrid.m37m.bed
# DATA_DIR=/dsde/data/deep/vqsr/tensors/big_mix_channels_last/
# BAM_FILE=/dsde/data/deep/vqsr/bams/chmi_chmi3_wgs1_g94794_bamout_all_calls.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/chmi_chmi3_wgs1_g94794_all_calls.vcf.gz


# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, exome nex1
# DOWNSAMPLE_SNPS=0.1
# TRAIN_VCF=/dsde/data/hybrid.m37m.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/hybrid.m37m.bed
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_mix_channels_first/
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_c1963_chmi_chmi3_nex1_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_c1963_chmi_chmi3_nex1.vcf.gz


# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, exome nex2
# DOWNSAMPLE_SNPS=0.1
# TRAIN_VCF=/dsde/data/hybrid.m37m.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/hybrid.m37m.bed
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_mix_channels_first/
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_c1963_chmi_chmi3_nex2_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_c1963_chmi_chmi3_nex2.vcf.gz


# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, replicate 4
# DOWNSAMPLE_SNPS=0.05 #0.005
# DOWNSAMPLE_INDELS=0.2 #0.03
# DOWNSAMPLE_NOT_SNPS=0.3
# DOWNSAMPLE_NOT_INDELS=0.15
# TRAIN_VCF=/dsde/data/hybrid.m37m.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/hybrid.m37m.bed
# DATA_DIR=/dsde/data/deep/vqsr/tensors/big_mix_channels_last/
# BAM_FILE=/dsde/data/deep/vqsr/bams/chmi_chmi3_wgs4_g94794_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/chmi_chmi3_wgs4_g94794.vcf.gz.recalibrated.vcf.gz


# From Palantir Wiki gold standard datasets: NA12878, PCRfree, 2x150, 30x coverage target
# MODE=write_tensors
# DOWNSAMPLE_SNPS=0.0075
# DOWNSAMPLE_INDELS=0.075
# DATA_DIR=/dsde/data/deep/vqsr/tensors/big_mix_channels_last/
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/recalibrated.vcf.gz
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_solexa_269365_bamout_no_trim.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed


# Project G94982 snapshot NA12878 with Mapping Quality
# DOWNSAMPLE_SNPS=0.01
# DOWNSAMPLE_INDELS=0.2
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_balanced_channels_last/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed


# Project G76270 NA12878 
# DOWNSAMPLE_SNPS=0.0045
# DOWNSAMPLE_INDELS=0.075
# DOWNSAMPLE_NOT_SNPS=0.5
# DOWNSAMPLE_NOT_INDELS=0.15
DATA_DIR=/dsde/data/deep/vqsr/tensors/g76270_na12878_pcrplus_channels_last/
TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g76270_na12878_pcrplus.vcf.gz
BAM_FILE=/dsde/data/deep/vqsr/bams/g76270_na12878_pcrplus_bamout.bam
BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed


# Project G96830 snapshot NA12878 with Mapping Quality HG38 reference
# DOWNSAMPLE_SNPS=0.0075
# DOWNSAMPLE_INDELS=0.03
# DOWNSAMPLE_NOT_SNPS=1.0
# DOWNSAMPLE_NOT_INDELS=0.9
# DATA_DIR=/dsde/data/deep/vqsr/tensors/big_mix_channels_last/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g96830_na12878_hg38_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g96830_na12878_hg38.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed


# Project G94982 snapshot NA12878 HG38 Tensors
# DOWNSAMPLE_SNPS=0.0075
# DOWNSAMPLE_INDELS=0.075
# DOWNSAMPLE_NOT_SNPS=0.8
# DOWNSAMPLE_NOT_INDELS=1.0
# DATA_DIR=/dsde/data/deep/vqsr/tensors/big_mix_channels_last/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g94982_hg38_na12878.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g94982_hg38_na12878_bamout.bam
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed


# Project G94982 snapshot NA12878 with pairs
# WINDOW_SIZE=256
# TENSOR_MAP=read_tensor
# MODE=write_paired_read_tensors
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_paired_read/
# BAM_FILE=/seq/picard_aggregation/G94982/NA12878/current/NA12878.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz


# Project G94982 calling tensors
# MODE=write_calling_tensors
# DOWNSAMPLE_SNPS=0.5
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_calling_tensors_variant_sort_many_snps/
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz


# Project G94982 calling tensors 1D
# MODE=write_calling_tensors
# DOWNSAMPLE_SNPS=0.0015
# DOWNSAMPLE_INDELS=0.1
# DOWNSAMPLE_HOMOZYGOUS=0.03
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_channels_first_calling_small/
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz


# Project G94982 pileup tensors
# MODE=write_pileup_filter_tensors
# TENSOR_MAP=2d_mapping_quality
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_pileup_filter/
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz


# Project G94982 1D gnomAD tensors
# MODE=write_tensors_gnomad_annotations_per_allele_1d
# TENSOR_MAP=1d_annotations
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_gnomad_normed_hail_1d4/
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz


# Project G94982 snapshot from Palantir Wiki gold standard datasets: NA12878, PCRfree, 2x151
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_platinum/
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes_confident_regions.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_na12878.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz


# Project G94982 snapshot from Palantir Wiki gold standard datasets: NA12878, PCRfree, 2x151, Raw BAM
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_bam_test/
# BAM_FILE=/seq/picard_aggregation/G94982/NA12878/current/NA12878.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz


# Project G76270 NA12878
# TENSOR_MAP=2d_mapping_quality
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g76270_maddy_/
# BAM_FILE=/dsde/data/deep/vqsr/bams/g76270_na12878.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/working/mduran/STR_calling_model/vcf_files/PCRplus_WGS_CONSERVATIVE.vcf.gz


# Project G71602 sample NA12877 The Husband from Utah CEPH CEU
# DOWNSAMPLE_SNPS=0.015
# DOWNSAMPLE_INDELS=0.1
# DOWNSAMPLE_NOT_SNPS=0.5
# DOWNSAMPLE_NOT_INDELS=0.15
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12877_bamout.bam 
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g71602_na12877_recalibrated.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes_confident_regions.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_genomes_NA12877.genome.vcf.gz
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g71602_na12877_balanced_channels_last/


# Project D5800 exome sample NA12877 The Husband from Utah CEPH CEU
# DOWNSAMPLE_SNPS=0.1
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_mix_channels_first/
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_d5800_na12877_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_d5800_na12877.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes_confident_regions.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_genomes_NA12877.genome.vcf.gz


# Project C1925 exome sample NA12877 The Husband from Utah CEPH CEU
# DOWNSAMPLE_SNPS=0.1
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_mix_channels_first/
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_c1925_na12877_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_c1925_na12877.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes_confident_regions.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_genomes_NA12877.genome.vcf.gz


# Project solexa_269364 sample NA12877 The Husband from Utah CEPH CEU
# DOWNSAMPLE_SNPS=0.015
# DOWNSAMPLE_INDELS=0.1
# DOWNSAMPLE_NOT_SNPS=0.5
# DOWNSAMPLE_NOT_INDELS=0.2
# DATA_DIR=/dsde/data/deep/vqsr/tensors/solexa_269364_na12877_balanced_channels_last/
# BAM_FILE=/dsde/data/deep/vqsr/bams/solexa_269364_na12877_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/solexa_269364_na12877.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes_confident_regions.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_genomes_NA12877.genome.vcf.gz


# Project G71602 sample NA12877 The Husband from Utah CEPH CEU 1D
# MODE=write_dna_tensors
# DOWNSAMPLE_INDELS=0.3
# TENSOR_MAP=1d_annotations
# DATA_DIR=/dsde/data/deep/vqsr/tensors/mix_id/
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12877_bamout.bam 
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g71602_na12877_recalibrated.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes_confident_regions.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_genomes_NA12877.genome.vcf.gz


# Project G100862 NA12892 Mother of NA12878, Utah CEPH CEU
# TENSOR_MAP=2d_mapping_quality
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g100862_na12892_gzip_mq/
# BAM_FILE=/dsde/data/deep/vqsr/bams/g100862_na12892_bamout.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes_confident_regions.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_genomes_na12892.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g100862_na12892.vcf.gz


# Project G100862 NA12891 Father of NA12878, Utah CEPH CEU
# TENSOR_MAP=2d_mapping_quality
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g100862_na12891_gzip_mq/
# BAM_FILE=/dsde/data/deep/vqsr/bams/g100862_na12891_bamout.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes_confident_regions.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_genomes_na12891.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g100862_na12891.vcf.gz


# DOWNSAMPLE_SNPS=0.02
# DOWNSAMPLE_INDELS=0.25
# TENSOR_MAP=2d_mapping_quality
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g100862_na12891_het_sort_no_soft/
# BAM_FILE=/dsde/data/deep/vqsr/bams/g100862_na12891_bamout.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes_confident_regions.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_genomes_na12891.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g100862_na12891.vcf.gz


# Project G94882 NA12878, 10X chemistry, Truth VCF from NIST
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_10x_bamout.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_10x.vcf.gz


# Project G94882 Ashkenazim Trio Father NA24149, 10X chemistry, Truth VCF from NIST
# TENSOR_MAP=2d_mapping_quality
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94882_na24149_10x_gzip_mq/
# BAM_FILE=/dsde/data/deep/vqsr/bams/NA24149_10X_bamout.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/hg003_intersect_hg001.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/hg003_na24149.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/NA24149_10X.bam.raw.vcf.gz


# Project G94882 Ashkenazim Trio Mother NA24143, 10X chemistry, Truth VCF from NIST
# DOWNSAMPLE_SNPS=0.2
# DOWNSAMPLE_INDELS=0.25
# TENSOR_MAP=2d_mapping_quality
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94882_na24143_10x_het_sort/
# BAM_FILE=/dsde/data/deep/vqsr/bams/NA24143_10X_bamout.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/HG004_GRCh37_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/hg004_na24143.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/NA24143_10X.bam.raw.recalibrated.vcf.gz


# Project D5800 NA12878 Exome long reads
# DOWNSAMPLE_SNPS=0.1
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_mix_channels_first/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_d5800_na12878.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_d5800_na12878_bamout.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed


# Project D5301 NA12878 Exome short reads
# DOWNSAMPLE_SNPS=0.1
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_mix_channels_first/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_d5301_na12878.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_d5301_na12878_bamout.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed


# Project D5227 NA12878 Exome control well
# DOWNSAMPLE_SNPS=0.1
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_mix_channels_first/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_na12878_nexpond_392292.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_na12878_nexpond_392292_all_calls.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed


# PrecisionFDA Hidden Treasure Challenge
# DOWNSAMPLE_SNPS=1.0
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_precision_fda_na12878_channels_first/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_hidden_treasure_na12878.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_hidden_treasure_na12878_bamout.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed


# Maddy Exome Tests
# READ_LIMIT=256
# DOWNSAMPLE_SNPS=0.2
# TENSOR_MAP=2d_mapping_quality
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_mix_str_model/
# BAM_FILE=/dsde/working/mduran/STR_calling_model/extra_exomes/bam_files/521582_MODEL.bam
# NEGATIVE_VCF=/dsde/working/mduran/STR_calling_model/extra_exomes/vcf_files/521582_MODEL.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz

. /broad/software/scripts/useuse

# use Java-1.8
# INTERVAL=/seq/references/Homo_sapiens_assembly19/v1/variant_calling/wgs_calling_regions.v1.interval_list
# PICARD=/seq/software/picard/current/bin/picard.jar

# Uncomment to create interval list with max interval length 10Mbp
# $ECHO java -jar $PICARD IntervalListTools BREAK_BANDS_AT_MULTIPLES_OF=10000000 I=$INTERVAL O=wgs_10m_split_genome.interval_list

# Remove annotation and format fields from a vcf (truth vcfs only need the sites)
# use Tabix
# use .bcftools-1.3.1
# $ECHO bcftools annotate -x INFO,FMT na24143_nist_giab.vcf > na24143_giab.vcf
# $ECHO sed 's/chr//' na24143_giab.vcf > na24143_giab_b37.vcf

# $ECHO bgzip -c platinum_na12878.vcf > platinum_na12878.vcf.gz
# $ECHO tabix -p vcf platinum_na12878.vcf.gz

use GridEngine8
use Anaconda

while read line; do
   if [[ ${line:0:1} != '@' ]]; then
      CHROM=$(echo $line | cut -d' ' -f1)
      START=$(echo $line | cut -d' ' -f2)
      END=$(echo $line | cut -d' ' -f3)
      $ECHO ./scripts/qsub_standard.sh -x -m 6g -n python_job'_'$CHROM'_'$START'_'$END \
            python training_data.py $MODE \
            --chrom $CHROM \
            --end_pos $END \
            --$CHANNEL_ORDER \
            --start_pos $START \
            --samples 10000000 \
            --bed_file $BED_FILE \
            --data_dir $DATA_DIR \
            --bam_file $BAM_FILE \
            --train_vcf $TRAIN_VCF \
            --tensor_map $TENSOR_MAP \
            --read_limit $READ_LIMIT \
            --window_size $WINDOW_SIZE \
            --reference_fasta $REFERENCE \
            --negative_vcf $NEGATIVE_VCF \
            --annotation_set $ANNOTATION_SET \
            --downsample_snps $DOWNSAMPLE_SNPS \
            --downsample_indels $DOWNSAMPLE_INDELS \
            --downsample_not_snps $DOWNSAMPLE_NOT_SNPS \
            --downsample_not_indels $DOWNSAMPLE_NOT_INDELS \
            --downsample_homozygous $DOWNSAMPLE_HOMOZYGOUS \
            --base_quality_mode $BASE_QUALITY_MODE
   fi
done < <(cat $SPLIT_INTERVALS)
