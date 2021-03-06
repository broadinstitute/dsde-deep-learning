#!/bin/bash
ECHO=

MODE=write_tensors

# Tensor definition
TENSOR_MAP=read_tensor
TENSOR_TYPES=read_tensor
READ_SORT=base
READ_LIMIT=128
WINDOW_SIZE=128
BASE_QUALITY_MODE=phot
CHANNEL_ORDER=channels_first
ANNOTATION_SET=best_practices
SAMPLE_NAME=NA12878
LABEL_TYPE=label_sites

# Downsample certain types of variant
DOWNSAMPLE_SNPS=0.015
DOWNSAMPLE_INDELS=0.1
DOWNSAMPLE_NOT_SNPS=1.0
DOWNSAMPLE_NOT_INDELS=1.0
DOWNSAMPLE_HOMOZYGOUS=1.0
DOWNSAMPLE_REFERENCE=0.001

# Data splitting
VALID_RATIO=0.1
VALID_CONTIGS="18 19 chr18 chr19"
TEST_RATIO=0.2
TEST_CONTIGS="2 20 chr2 chr20"

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

# Clinical NA24385 Ashkenazi son
# DOWNSAMPLE_SNPS=0.004
# DOWNSAMPLE_INDELS=0.025
# CHANNEL_ORDER=channels_last
# SAMPLE_NAME=SM-G947H
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g947h_na24385_channels_last/
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947h_o1d1v1_na24385_bamout.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24385_hg38_v3_3_2.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947h_o1d1v1_na24385_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG002_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC-SOLIDgatkHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

# Clinical NA24385 Ashkenazi son
# DOWNSAMPLE_SNPS=0.01
# DOWNSAMPLE_INDELS=0.1
# SAMPLE_NAME=SM-G947H
# WINDOW_SIZE=128
# TENSOR_TYPES="read_tensor paired_reads reads_only reads_reference"
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_g947h_na24385_ro_rr_rt_pr/
# BAM_FILE=/dsde/data/deep/vqsr/bams/nova_g947h_na24385_bamout.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24385_hg38_v3_3_2.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/nova_g947h_na24385_cnn_scored.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG002_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC-SOLIDgatkHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed


# DOWNSAMPLE_SNPS=0.008
# DOWNSAMPLE_INDELS=0.03
# SAMPLE_NAME=SM-G947H
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_g947h_na24385_hg38/
# BAM_FILE=/dsde/data/deep/vqsr/bams/nova_g947h_na24385_bamout.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24385_hg38_v3_3_2.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/nova_g947h_na24385_cnn_scored.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG002_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC-SOLIDgatkHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

# Clinical NA24385 Ashkenazi son
# SAMPLE_NAME=SM-G947T
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g947_paired_read2/
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947t_o2d1v1_na24385_merged.vcf.gz 
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947t_o2d1v1_na24385_bamout.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24385_hg38_v3_3_2.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG002_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC-SOLIDgatkHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

# NovaSeq NA24385 Ashkenazi son
# SAMPLE_NAME=SM-G947T
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_g947t_na24385_hg38_new/
# BAM_FILE=/dsde/data/deep/vqsr/bams/nova_g947t_na24385_bamout.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24385_hg38_v3_3_2.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/nova_g947t_na24385_cnn_scored.vcf.gz 
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG002_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC-SOLIDgatkHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

# Clinical NA24385 Ashkenazi son
# SAMPLE_NAME=SM-G9486
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g9486_na24385_hg38/
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g9486_na24385_o1d2v1_cnn_scored.vcf.gz 
# BAM_FILE=/dsde/data/deep/vqsr/bams/g9486_na24385_o1d2v1_bamout.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24385_hg38_v3_3_2.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG002_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC-SOLIDgatkHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed


# Clinical 24149 Ashkenazi Father HG003
# SAMPLE_NAME=SM-G947I
# TENSOR_TYPES="read_tensor paired_reads reads_only reads_reference"
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g947i_na24149_hg38_ro_rr_rt_pr/
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947i_o2d1v1_na24149_bamout.bam 
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24149_hg38_v3_3_2.vcf.gz 
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947i_o2d1v1_na24149_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG003_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

# Nova 24149 Ashkenazi Father HG003
# SAMPLE_NAME=SM-G947I
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# BAM_FILE=/dsde/data/deep/vqsr/bams/nova_g947i_na24149_bamout.bam 
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24149_hg38_v3_3_2.vcf.gz 
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/nova_g947i_na24149_cnn_scored.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG003_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

# DOWNSAMPLE_SNPS=0.004
# DOWNSAMPLE_INDELS=0.025
# CHANNEL_ORDER=channels_last
# SAMPLE_NAME=SM-G947U
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947u_o1d1v2_na24149_bamout.bam 
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24149_hg38_v3_3_2.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947u_o1d1v2_na24149_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG003_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

# Clinical 24149 Ashkenazi Father HG003
# SAMPLE_NAME=SM-G947J
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947j_o1d2v1_na24149_bamout.bam 
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24149_hg38_v3_3_2.vcf.gz 
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947j_o1d2v1_na24149_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG003_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

# NovaSeq Clinical 24149 Ashkenazi Father HG003
# SAMPLE_NAME=SM-G947J
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# BAM_FILE=/dsde/data/deep/vqsr/bams/nova_g947j_na24149_bamout.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24149_hg38_v3_3_2.vcf.gz 
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/nova_g947j_na24149_cnn_scored.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG003_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed



# Clinical 24143 Ashkenazi Mother HG004
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g947v_o1d1v1_na24143_ws256_rl192_cf/
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947v_o1d1v1_na24143_bamout.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24143_hg38_v3_3_2.vcf.gz 
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947v_o1d1v1_na24143_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG004_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

# Clinical 24143 Ashkenazi Mother HG004
# SAMPLE_NAME=SM-G947K
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947k_o2d1v1_na24143_bamout.bam 
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24143_hg38_v3_3_2.vcf.gz 
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947k_o2d1v1_na24143_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG004_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

# Clinical 24143 Ashkenazi Mother HG004
# SAMPLE_NAME=SM-G947K
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# BAM_FILE=/dsde/data/deep/vqsr/bams/nova_g947k_na24143_bamout.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24143_hg38_v3_3_2.vcf.gz 
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/nova_g947k_na24143_filtered.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG004_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

# Clinical 24143 Ashkenazi Mother HG004
# SAMPLE_NAME=SM-G947W
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947w_o1d2v4_na24143_bamout.bam 
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/giab_na24143_hg38_v3_3_2.vcf.gz 
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947w_o1d2v4_na24143_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG004_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed


# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, replicate 1
# DOWNSAMPLE_SNPS=0.005
# DOWNSAMPLE_INDELS=0.05
# DOWNSAMPLE_NOT_SNPS=0.25
# DOWNSAMPLE_NOT_INDELS=0.4
# SAMPLE_NAME=CHMI_CHMI3_WGS1
# BED_FILE=/dsde/working/sam/CHM-eval.kit/full.37m.bed
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94794_wgs1_hg19/
# TRAIN_VCF=/dsde/working/sam/CHM-eval.kit/full.37m.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/chmi_chmi3_wgs1_g94794_bamout_all_calls.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/chmi_chmi3_wgs1_g94794_all_calls.vcf.gz

# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, replicate 4
# DOWNSAMPLE_SNPS=0.05 #0.005
# DOWNSAMPLE_INDELS=0.2 #0.03
# DOWNSAMPLE_NOT_SNPS=0.3
# DOWNSAMPLE_NOT_INDELS=0.15
# TRAIN_VCF=/dsde/data/hybrid.m37m.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/hybrid.m37m.bed
# DATA_DIR=/dsde/data/deep/vqsr/tensors/big_mix_channels_last/
# BAM_FILE=/dsde/data/deep/vqsr/bams/chmi_chmi3_wgs4_g94794_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/chmi_chmi3_wgs4_g94794.vcf.gz

# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, replicate 4
# DOWNSAMPLE_SNPS=0.15
# MODE=write_tensors_gnomad_annotations_1d
# TENSOR_MAP=reference
# DATA_DIR=/dsde/data/deep/vqsr/tensors/syndip_wgs4_hg19_gnomad_anno/
# TRAIN_VCF=/dsde/working/sam/CHM-eval.kit/full.37m.vcf.gz
# BED_FILE=/dsde/working/sam/CHM-eval.kit/full.37m.bed
# BAM_FILE=/dsde/data/deep/vqsr/bams/chmi_chmi3_wgs4_g94794_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/chmi_chmi3_wgs4_g94794.vcf.gz~.

# DOWNSAMPLE_SNPS=0.75
# DOWNSAMPLE_NOT_SNPS=0.45
# MODE=write_tensors_gnomad_annotations_1d
# TENSOR_MAP=reference
# DATA_DIR=/dsde/data/deep/vqsr/tensors/syndip_hg19_gnomad_anno_rf_labels_chrom1/
# TRAIN_VCF=/dsde/working/sam/CHM-eval.kit/full.37m.vcf.gz
# BED_FILE=/dsde/working/sam/CHM-eval.kit/full.37m.bed
# BAM_FILE=/dsde/data/deep/vqsr/bams/chmi_chmi3_wgs1_g94794_bamout_all_calls.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/chmi_chmi3_wgs1_g94794_all_calls.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/chrom1_10m_split.interval_list

# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, exome nex1
# DOWNSAMPLE_SNPS=0.2
# DOWNSAMPLE_INDELS=1.0
# SAMPLE_NAME=CHMI_CHMI3_Nex1
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_syndip_mix2/
# TRAIN_VCF=/dsde/working/sam/CHM-eval.kit/full.37m.vcf.gz
# BED_FILE=/dsde/working/sam/CHM-eval.kit/full.37m.bed
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_c1963_chmi_chmi3_nex1_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_c1963_chmi_chmi3_nex1.vcf.gz

# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, exome nex2
# DOWNSAMPLE_SNPS=0.2
# DOWNSAMPLE_INDELS=1.0
# SAMPLE_NAME=CHMI_CHMI3_Nex2
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_syndip_mix2/
# TRAIN_VCF=/dsde/working/sam/CHM-eval.kit/full.37m.vcf.gz
# BED_FILE=/dsde/working/sam/CHM-eval.kit/full.37m.bed
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_c1963_chmi_chmi3_nex2_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_c1963_chmi_chmi3_nex2.vcf.gz

# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, exome nex3
# DOWNSAMPLE_SNPS=0.2
# DOWNSAMPLE_INDELS=1.0
# SAMPLE_NAME=CHMI_CHMI3_Nex3
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_syndip_mix2/
# TRAIN_VCF=/dsde/working/sam/CHM-eval.kit/full.37m.vcf.gz
# BED_FILE=/dsde/working/sam/CHM-eval.kit/full.37m.bed
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_c1963_chmi_chmi3_nex3_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_c1963_chmi_chmi3_nex3.vcf.gz

# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, exome nex4
# DOWNSAMPLE_SNPS=0.2
# DOWNSAMPLE_INDELS=1.0
# SAMPLE_NAME=CHMI_CHMI3_Nex4
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_syndip_mix2/
# TRAIN_VCF=/dsde/working/sam/CHM-eval.kit/full.37m.vcf.gz
# BED_FILE=/dsde/working/sam/CHM-eval.kit/full.37m.bed
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_c1963_chmi_chmi3_nex4_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_c1963_chmi_chmi3_nex4.vcf.gz

# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, WGS1 HG38
# DOWNSAMPLE_SNPS=0.02 #0.005
# DOWNSAMPLE_INDELS=0.2 #0.03
# DOWNSAMPLE_NOT_SNPS=0.3
# DOWNSAMPLE_NOT_INDELS=0.15
# SAMPLE_NAME=CHMI_CHMI3_WGS1
# TENSOR_TYPES="read_tensor paired_reads reads_only reads_reference"
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94794_wgs1_hg38_ro_rr_rt_pr/
# TRAIN_VCF=/dsde/working/sam/CHM-eval.kit/full.38.vcf.gz
# BED_FILE=/dsde/working/sam/CHM-eval.kit/full.38.bed
# BAM_FILE=/dsde/data/deep/vqsr/bams/g94794_chm_wgs1_hg38_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g94794_chm_wgs1_hg38_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta

# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, WGS2 HG38
# DOWNSAMPLE_INDELS=0.25
# SAMPLE_NAME=CHMI_CHMI3_WGS2
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94794_wgs2_hg38_full/
# TRAIN_VCF=/dsde/data/palantir/CHM-eval.kit/hybrid.m38.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/full.38.bed
# BAM_FILE=/dsde/data/deep/vqsr/bams/g94794_chm_wgs2_hg38_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g94794_chm_wgs2_hg38_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta

# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, WGS3 HG38
# DOWNSAMPLE_INDELS=0.4
# SAMPLE_NAME=CHMI_CHMI3_WGS3
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94794_wgs3_hg38_full/
# TRAIN_VCF=/dsde/data/palantir/CHM-eval.kit/hybrid.m38.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/full.38.bed
# BAM_FILE=/dsde/data/deep/vqsr/bams/g94794_chm_wgs3_hg38_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g94794_chm_wgs3_hg38_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta

# The Haploid Mix, Synthetic Diploid, CHMI and CHMI3, WGS4 HG38
# DOWNSAMPLE_INDELS=0.4
# SAMPLE_NAME=CHMI_CHMI3_WGS4
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94794_wgs4_hg38_full/
# TRAIN_VCF=/dsde/data/palantir/CHM-eval.kit/hybrid.m38.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/full.38.bed
# BAM_FILE=/dsde/data/deep/vqsr/bams/g94794_chm_wgs4_hg38_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g94794_chm_wgs4_hg38_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta


# From Palantir Wiki gold standard datasets: NA12878, PCRfree, 2x150, 30x coverage target
# DOWNSAMPLE_SNPS=0.01
# DOWNSAMPLE_INDELS=0.2
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_solexa_269365_bamout.raw.snps.indels.vcf.gz
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_solexa_269365_bamout_no_trim.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

# Project G94982 snapshot NA12878
# DOWNSAMPLE_SNPS=0.01
# DOWNSAMPLE_INDELS=0.2
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_na12878_realign/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_g94982_bamout.raw.snps.indels.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

# Project G94982 snapshot NA12878
# DOWNSAMPLE_SNPS=0.01
# DOWNSAMPLE_INDELS=0.2
# TENSOR_TYPES="read_tensor paired_reads reads_only reads_reference"
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_na12878_rorr_rtpr_ws128/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_g94982_bamout.raw.snps.indels.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed


# Project G94982 snapshot NA12878
# DOWNSAMPLE_SNPS=0.01
# DOWNSAMPLE_INDELS=0.2
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_na12878_no_realign/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# BAM_FILE=/seq/picard_aggregation/G94982/NA12878/current/NA12878.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_g94982_bamout.raw.snps.indels.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

# Project G94982 snapshot NA12878 Platinum Genomes 2017 as truth
# DOWNSAMPLE_SNPS=0.02
# DOWNSAMPLE_INDELS=0.1
# DOWNSAMPLE_NOT_SNPS=0.2
# DOWNSAMPLE_NOT_INDELS=0.15
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_na12878_hg19_platinum_fix/
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_genomes/pg_hg19_na12878.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_g94982_bamout.raw.snps.indels.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes/ConfidentRegions_pg_b37_2017.bed

# Project G94982 snapshot NA12878 Platinum Genomes GiaB hybrid 2017 as truth
# DOWNSAMPLE_SNPS=0.02
# DOWNSAMPLE_INDELS=0.15
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_na12878_hg19_platinum_giab_hybrid_fix/
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes/pg_hg19_giab_hybrid.bed
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_g94982_bamout.raw.snps.indels.vcf.gz
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_genomes/pg_hg19_giab_hybrid_na12878.vcf.gz

# Project G76270 NA12878 
# DOWNSAMPLE_SNPS=0.01
# DOWNSAMPLE_INDELS=0.2
# DOWNSAMPLE_NOT_SNPS=1.0
# DOWNSAMPLE_NOT_INDELS=0.4
# DATA_DIR=/dsde/data/deep/vqsr/tensors/mix_hg19_na12878/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g76270_na12878_pcrplus.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g76270_na12878_pcrplus_bamout.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

# Project G96830 snapshot NA12878 with Mapping Quality HG38 reference
# DOWNSAMPLE_SNPS=0.0075
# DOWNSAMPLE_INDELS=0.03
# DOWNSAMPLE_NOT_SNPS=1.0
# DOWNSAMPLE_NOT_INDELS=0.9
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g96830_na12878_hg38_channels_first/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g96830_na12878_hg38_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g96830_na12878_hg38.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Clinical NA12878 Intrarun2
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g947_mix_allele_labelled_anno_het0/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947n_intrarun2_na12878_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947n_intrarun2_na12878_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Clinical NA12878 100ng
# DOWNSAMPLE_SNPS=0.003
# DOWNSAMPLE_INDELS=0.025
# DOWNSAMPLE_NOT_SNPS=0.5
# CHANNEL_ORDER=channels_last
# SAMPLE_NAME=SM-G947Q
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947q_lod_100ng_na12878_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947q_lod_100ng_na12878_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Clinical NA12878 g947p 250ng DNA
# SAMPLE_NAME=SM-G947P
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947p_na12878_250ng_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947p_na12878_250ng_cnn_scored.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Clinical NA12878 500ng DNA
# DOWNSAMPLE_SNPS=0.003
# DOWNSAMPLE_INDELS=0.025
# DOWNSAMPLE_NOT_SNPS=0.5
# CHANNEL_ORDER=channels_last
# SAMPLE_NAME=SM-G9482
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g9482_lod_500ng_na12878_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g9482_lod_500ng_na12878_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Clinical NA12878 750ng DNA
# SAMPLE_NAME=SM-G9483
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g9483_lod_750ng_na12878_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g9483_lod_750ng_na12878_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Clinical NA12878 1ug
# WINDOW_SIZE=256
# DOWNSAMPLE_SNPS=0.003
# DOWNSAMPLE_INDELS=0.025
# DOWNSAMPLE_NOT_SNPS=0.5
# CHANNEL_ORDER=channels_last
# SAMPLE_NAME=SM-G9481
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g94781_lod_1ug_na12878_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g94781_lod_1ug_na12878_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Clinical NA12878 g947l
# SAMPLE_NAME=SM-G947L
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g947_mix_site_labelled/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947l_o1d1v1_na12878_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947l_o1d1v1_na12878_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Nova NA12878 g947l
# SAMPLE_NAME=SM-G947L
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_mix/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/nova_g947l_na12878_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/nova_g947l_na12878_cnn_scored.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Clinical NA12878 g947m
# DOWNSAMPLE_SNPS=0.003
# DOWNSAMPLE_INDELS=0.025
# DOWNSAMPLE_NOT_SNPS=0.5
# SAMPLE_NAME=SM-G947M
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g947m_na12878/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947m_o1d2v1_na12878_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947m_o1d2v1_na12878_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Novaseq NA12878 g947m
# SAMPLE_NAME=SM-G947M
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# BAM_FILE=/dsde/data/deep/vqsr/bams/nova_g947m_na12878_bamout.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/nova_g947m_na12878_cnn_scored.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Nova NA12878 g947n
# SAMPLE_NAME=SM-G947N
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# BAM_FILE=/dsde/data/deep/vqsr/bams/nova_g947n_na12878_bamout.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/nova_g947n_na12878_cnn_scored.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Clinical NA12878 g947o
# DOWNSAMPLE_SNPS=0.005
# DOWNSAMPLE_INDELS=0.025
# CHANNEL_ORDER=channels_last
# SAMPLE_NAME=SM-G947O
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947o_intrarun4_na12878_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947o_intrarun4_na12878_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed


# Clinical NA12878 g947x
# DOWNSAMPLE_SNPS=0.005
# DOWNSAMPLE_INDELS=0.025
# CHANNEL_ORDER=channels_last
# SAMPLE_NAME=SM-G947X
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947x_o2d1v1_na12878_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947x_o2d1v1_na12878_cnn_scored.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Nova NA12878 g947x
DOWNSAMPLE_SNPS=0.01
DOWNSAMPLE_INDELS=0.1
SAMPLE_NAME=SM-G947X
TENSOR_TYPES="read_tensor paired_reads reads_only reads_reference"
DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_g947x_na12878_ro_rr_rt_pr/
BAM_FILE=/dsde/data/deep/vqsr/bams/nova_g947x_na12878_bamout.bam
TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/nova_g947x_na12878_cnn_scored.vcf.gz
SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Clinical NA12878 g947z
# WINDOW_SIZE=256
# DOWNSAMPLE_SNPS=0.003
# DOWNSAMPLE_INDELS=0.025
# DOWNSAMPLE_NOT_SNPS=0.5
# SAMPLE_NAME=SM-G947Z
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g947z_intrarun3_na12878_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g947z_intrarun3_na12878_hc4_merged.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Clinical NA12878 g947z
# DOWNSAMPLE_SNPS=0.003
# DOWNSAMPLE_INDELS=0.025
# DOWNSAMPLE_NOT_SNPS=0.5
# CHANNEL_ORDER=channels_last
# SAMPLE_NAME=SM-G947Z
# DATA_DIR=/dsde/data/deep/vqsr/tensors/nova_hiseq_mix/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/nova_g947z_na12878_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/nova_g947z_na12878_filtered.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed



# Illumina Made NA12878 S1
#DOWNSAMPLE_SNPS=0.008
#DOWNSAMPLE_INDELS=0.07
#DOWNSAMPLE_NOT_SNPS=0.8
#CHANNEL_ORDER=channels_last
# DATA_DIR=/dsde/data/deep/vqsr/tensors/illumina_na12878_s1_site_label/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/illumina_na12878_s1_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/illumina_na12878_s1.vcf.gz
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Illumina Made NA12878 S1 with platinum truth
# TEST_CONTIGS="chr2 chr20"
# DATA_DIR=/dsde/data/deep/vqsr/tensors/illumina_na12878_s1_platinum/
# BAM_FILE=/dsde/data/deep/vqsr/bams/illumina_na12878_s1_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/illumina_na12878_s1.vcf.gz
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinumgenomes_hg38.NA12878.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes_confident_regions_hg38.bed
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta

# Project G94982 snapshot NA12878 HG38 Tensors
# DOWNSAMPLE_SNPS=0.0075
# DOWNSAMPLE_INDELS=0.075
# DOWNSAMPLE_NOT_SNPS=0.8
# DOWNSAMPLE_NOT_INDELS=1.0
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_hg38_na12878/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g94982_hg38_na12878.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/g94982_hg38_na12878_bamout.bam
# SPLIT_INTERVALS=/dsde/data/deep/vqsr/beds/wgs_10m_split_genome_hg38.interval_list
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# BED_FILE=/dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

# Project G94982 calling tensors
# MODE=write_calling_tensors
# CHANNEL_ORDER=channels_last
# DOWNSAMPLE_SNPS=0.05
# DOWNSAMPLE_HOMOZYGOUS=0.2
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_calling_channels_last/
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_g94982_bamout.raw.snps.indels.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

# Project G94982 calling tensors 1D
# MODE=write_calling_tensors
# DOWNSAMPLE_SNPS=0.0015
# DOWNSAMPLE_INDELS=0.1
# DOWNSAMPLE_HOMOZYGOUS=0.03
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_channels_first_calling_small/
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_g94982_bamout.raw.snps.indels.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

# Project G94982 1D gnomAD tensors
# MODE=write_tensors_gnomad_annotations_per_allele_1d
# TENSOR_MAP=reference
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_gnomad_normed_hail_1d4/
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_g94982_bamout.raw.snps.indels.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

# Project G94982 1D gnomAD tensors
# MODE=write_tensors_gnomad_annotations_1d
# TENSOR_MAP=reference
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_g94982_bamout_no_trim.bam
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_gnomad_annos_rf_negatives/
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_g94982_bamout.raw.snps.indels.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

# Project G94982 1D gnomAD tensors
# MODE=write_tensors_gnomad_annotations_1d
# TENSOR_MAP=reference
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_gnomad_annos_rf_negatives/
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_solexa_269365_bamout.raw.snps.indels.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12878_solexa_269365_bamout_no_trim.bam

# Project G94982 snapshot from Palantir Wiki gold standard datasets: NA12878, PCRfree, 2x151, Raw BAM
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g94982_bam_test/
# BAM_FILE=/seq/picard_aggregation/G94982/NA12878/current/NA12878.bam
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/na12878_g94982_bamout.raw.snps.indels.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

# Project G76270 NA12878
# TENSOR_MAP=2d_mapping_quality
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g76270_maddy_/
# BAM_FILE=/dsde/data/deep/vqsr/bams/g76270_na12878.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/working/mduran/STR_calling_model/vcf_files/PCRplus_WGS_CONSERVATIVE.vcf.gz

# Project G71602 sample NA12877 The Husband from Utah CEPH CEU
# DOWNSAMPLE_SNPS=0.01
# DOWNSAMPLE_INDELS=0.1
# DOWNSAMPLE_NOT_SNPS=0.2
# DOWNSAMPLE_NOT_INDELS=0.15
# SAMPLE_NAME=NA12877
# BAM_FILE=/dsde/data/deep/vqsr/bams/na12877_bamout.bam 
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/g71602_na12877_hg19.vcf.gz
# DATA_DIR=/dsde/data/deep/vqsr/tensors/g71602_hg19_na12877_platinum2/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_genomes/pg_hg19_na12877.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes/pg_hg19_confident_regions.bed


# Project D5800 exome sample NA12877 The Husband from Utah CEPH CEU
# SAMPLE_NAME=NA12877
# DOWNSAMPLE_SNPS=0.4
# DOWNSAMPLE_INDELS=1.0
# CHANNEL_ORDER=channels_last
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_mix_channels_last2/
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_d5800_na12877_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_d5800_na12877.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/platinum_genomes_confident_regions.bed
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/platinum_genomes_NA12877.genome.vcf.gz

# Project C1925 exome sample NA12877 The Husband from Utah CEPH CEU
# SAMPLE_NAME=NA12877
# DOWNSAMPLE_SNPS=0.4
# DOWNSAMPLE_INDELS=1.0
# CHANNEL_ORDER=channels_last
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_mix_channels_last2/
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

# Project D5800 NA12878 Exome long reads
# DOWNSAMPLE_SNPS=0.4
# DOWNSAMPLE_INDELS=1.0
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_na12878_mix_hg19/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_d5800_na12878.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_d5800_na12878_bamout.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

# Project D5301 NA12878 Exome short reads
# DOWNSAMPLE_SNPS=0.4
# DOWNSAMPLE_INDELS=1.0
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_na12878_mix_hg19/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_d5301_na12878.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_d5301_na12878_bamout.bam
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

# Project D5227 NA12878 Exome control well
# DOWNSAMPLE_SNPS=0.4
# DOWNSAMPLE_INDELS=1.0
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_na12878_mix_hg19/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_na12878_nexpond_392292.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_na12878_nexpond_392292_all_calls.vcf.gz
# BED_FILE=/dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

# Project D5227 NA12878 Exome control well
# DOWNSAMPLE_SNPS=0.4
# DOWNSAMPLE_INDELS=1.0
# DATA_DIR=/dsde/data/deep/vqsr/tensors/exome_na12878_mix_hg19/
# TRAIN_VCF=/dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz
# BAM_FILE=/dsde/data/deep/vqsr/bams/exome_na12878_nexpond_560386_bamout.bam
# NEGATIVE_VCF=/dsde/data/deep/vqsr/vcfs/exome_na12878_nexpond_560386.vcf.gz
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
      $ECHO ./scripts/qsub_standard.sh -x -m 10g -n python_job'_'$CHROM'_'$START'_'$END \
            python training_data.py $MODE \
            --$LABEL_TYPE \
            --chrom $CHROM \
            --end_pos $END \
            --$CHANNEL_ORDER \
            --start_pos $START \
            --samples 10000000 \
            --bed_file $BED_FILE \
            --data_dir $DATA_DIR \
            --bam_file $BAM_FILE \
            --train_vcf $TRAIN_VCF \
            --read_sort $READ_SORT \
            --read_limit $READ_LIMIT \
            --test_ratio $TEST_RATIO \
            --tensor_map $TENSOR_MAP \
            --valid_ratio $VALID_RATIO \
            --window_size $WINDOW_SIZE \
            --sample_name $SAMPLE_NAME \
            --tensor_types $TENSOR_TYPES \
            --reference_fasta $REFERENCE \
            --negative_vcf $NEGATIVE_VCF \
            --test_contigs $TEST_CONTIGS \
            --valid_contigs $VALID_CONTIGS \
            --annotation_set $ANNOTATION_SET \
            --downsample_snps $DOWNSAMPLE_SNPS \
            --base_quality_mode $BASE_QUALITY_MODE \
            --downsample_indels $DOWNSAMPLE_INDELS \
            --downsample_not_snps $DOWNSAMPLE_NOT_SNPS \
            --downsample_reference $DOWNSAMPLE_REFERENCE \
            --downsample_not_indels $DOWNSAMPLE_NOT_INDELS \
            --downsample_homozygous $DOWNSAMPLE_HOMOZYGOUS 
   fi
done < <(cat $SPLIT_INTERVALS)
