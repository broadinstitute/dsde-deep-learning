#!/bin/bash

ECHO=echo

. /broad/software/scripts/useuse
use Java-1.8
use GridEngine8

GATKUnstable=/humgen/gsa-hpprojects/GATK/private_unstable_builds/GenomeAnalysisTK_latest_unstable.jar
GATK4=/dsde/working/sam/gatk/build/libs/gatk.jar
QUEUE=/humgen/gsa-hpprojects/Queue/private_unstable_builds/Queue_latest_unstable.jar
HAPLOTYPECALLER=./scripts/HaplotypeCallerSingleSampleScript.scala
VQSR=./scripts/VQSRScript.scala
REFERENCE=/seq/references/Homo_sapiens_assembly19/v1/Homo_sapiens_assembly19.fasta
TMPDIR=/broad/hptmp/`whoami`
INTERVAL=/seq/references/Homo_sapiens_assembly19/v1/variant_calling/wgs_calling_regions.v1.interval_list
PICARD=/seq/software/picard/current/bin/picard.jar
SCATTER=210


# HG002 BAM from NIST NA24385 Ashkenazi son
# BAM=/dsde/data/deep/vqsr/bams/HG002_NIST_150bp_50x.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/hg002_na24385_nist_bamout_all_calls.bam
# VCF=/dsde/data/deep/vqsr/vcfs/hg002_na24385_nist_150bp_50x_all_calls.vcf.gz

# HG002 BAM from clinical genomes NA24385 Ashkenazi son
# BAM=/dsde/data/deep/vqsr/bams/g947h_o1d1v1_na24385.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/g947h_o1d1v1_na24385_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/g947h_o1d1v1_na24385.vcf.gz
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# INTERVAL=/seq/references/Homo_sapiens_assembly38/v0/variant_calling/wgs_calling_regions.v1.interval_list 

# The synthetic diploid replicate 1
# BAM=/seq/picard_aggregation/G94794/CHMI_CHMI3_WGS1/v3/CHMI_CHMI3_WGS1.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/chmi_chmi3_wgs1_g94794_bamout_all_calls.bam
# VCF=/dsde/data/deep/vqsr/vcfs/chmi_chmi3_wgs1_g94794_all_calls.vcf.gz

# The synthetic diploid replicate 4
# BAM=/dsde/working/sam/cnn/CHMI_CHMI3_WGS4.bam
# BAMOUT=/dsde/working/sam/cnn/chmi_chmi3_wgs4_g94794_bamout_all_calls.bam
# VCF=/dsde/data/deep/vqsr/vcfs/chmi_chmi3_wgs4_g94794_all_calls.vcf.gz

# Good ole NA12878 CEPH CEU Utah wife
# BAM=/dsde/data/deep/vqsr/bams/NA12878_10X.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/na12878_10x_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/na12878_10x.vcf.gz

# Good ole NA12878 CEPH CEU Utah wife PCR plus v2 chemistry
# BAM=/dsde/data/deep/vqsr/bams/G76270_NA12878.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/g76270_na12878_pcrplus_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/g76270_na12878_pcrplus.vcf.gz

# NA12878 HG38
# BAM=/dsde/data/deep/vqsr/bams/G96830_NA12878_HG38.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/g96830_na12878_hg38_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/g96830_na12878_hg38.vcf.gz
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# INTERVAL=/seq/references/Homo_sapiens_assembly38/v0/variant_calling/wgs_calling_regions.v1.interval_list 

# Clinical project g47m NA12878 HG38
# BAM=/dsde/data/deep/vqsr/bams/g947m_o1d1v1_na12878.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/g947m_o1d1v1_na12878_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/g947m_o1d1v1_na12878.vcf.gz
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# INTERVAL=/seq/references/Homo_sapiens_assembly38/v0/variant_calling/wgs_calling_regions.v1.interval_list 


# NA12877 CEPH CEU Utah husband Solexa 269364
# BAM=/seq/picard_aggregation/LCSET-6205/Solexa-269364,_PCR-Free_pool/v1/Solexa-269364,_PCR-Free_pool.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/solexa_269364_na12877_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/solexa_269364_na12877.vcf.gz

# NA12877 CEPH CEU Utah husband
# BAM=/dsde/data/deep/vqsr/bams/NA12877.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/na12877_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/na12877.vcf.gz

# Project G100862 NA12892 BAM of Utah Mother of NA12878
# BAM=/seq/picard_aggregation/G100862/NA12892/v2/NA12892.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/g100862_na12892_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/g100862_na12892.vcf.gz

# Project G100862 NA12891 BAM of Utah Father of NA12878
# BAM=/seq/picard_aggregation/G100862/NA12891/v2/NA12891.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/g100862_na12891_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/g100862_na12891.vcf.gz

# Ashkenazi father
# BAM=/seq/picard_aggregation/G94882/NA24149_10X/current/NA24149_10X.bam
# BAMOUT=NA24149_10X.bamout.bam
# VCF=NA24149_10X.vcf

# Ashkenazi son
# BAM=/seq/picard_aggregation/G94882/NA24385_10X/v2/NA24385_10X.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/na24385_10x_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/na24385_10x.vcf.gz

# Exome Project D5800, NA12878, Long reads
# BAM=/seq/picard_aggregation/DEV-5800/Pond-391572,_NA12878_3/current/Pond-391572,_NA12878_3.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/exome_d5800_na12878_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/exome_d5800_na12878.vcf.gz
# INTERVAL=/seq/references/Homo_sapiens_assembly19/v1/variant_calling/exome_calling_regions.v1.interval_list

# Exome Project D5800, NA12891 Father, Long reads
# BAM=/seq/picard_aggregation/DEV-5800/Pond-391578,_NA12891_2/current/Pond-391578,_NA12891_2.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/exome_d5800_na12891_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/exome_d5800_na12891.vcf.gz
# INTERVAL=/seq/references/Homo_sapiens_assembly19/v1/variant_calling/exome_calling_regions.v1.interval_list

# Exome Project D5800, NA12877 Husband, Long reads
# BAM=/seq/picard_aggregation/DEV-5800/Pond-391569,_NA12877/current/Pond-391569,_NA12877.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/exome_d5800_na12877_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/exome_d5800_na12877.vcf.gz
# INTERVAL=/seq/references/Homo_sapiens_assembly19/v1/variant_calling/exome_calling_regions.v1.interval_list

# Exome Project D5301, NA12878, Short reads
# BAM=/seq/picard_aggregation/D5301/NexPond-377866/current/NexPond-377866.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/exome_d5301_na12878_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/exome_d5301_na12878.vcf.gz
# INTERVAL=/seq/references/Homo_sapiens_assembly19/v1/variant_calling/exome_calling_regions.v1.interval_list

# Exome Project C1925, NA12877
# BAM=/seq/picard_aggregation/C1925/NA12877/v3/NA12877.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/exome_c1925_na12877_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/exome_c1925_na12877.vcf.gz
# INTERVAL=/seq/references/Homo_sapiens_assembly19/v1/variant_calling/exome_calling_regions.v1.interval_list

# Exome Project C1963, CHMI_CHMI3 Nex 1
# BAM=/seq/picard_aggregation/C1963/CHMI_CHMI3_Nex1/v2/CHMI_CHMI3_Nex1.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/exome_c1963_chmi_chmi3_nex1_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/exome_c1963_chmi_chmi3_nex1.vcf.gz
# INTERVAL=/seq/references/Homo_sapiens_assembly19/v1/variant_calling/exome_calling_regions.v1.interval_list


# Exome Project C1963, CHMI_CHMI3 Nex 2
# BAM=/seq/picard_aggregation/C1963/CHMI_CHMI3_Nex2/v1/CHMI_CHMI3_Nex2.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/exome_c1963_chmi_chmi3_nex2_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/exome_c1963_chmi_chmi3_nex2.vcf.gz
# INTERVAL=/seq/references/Homo_sapiens_assembly19/v1/variant_calling/exome_calling_regions.v1.interval_list


# Exome Project D5301, NA12891 Father, Short reads
# BAM=/seq/picard_aggregation/D5301/NexPond-377873/current/NexPond-377873.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/exome_d5301_na12891_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/exome_d5301_na12891.vcf.gz
# INTERVAL=/seq/references/Homo_sapiens_assembly19/v1/variant_calling/exome_calling_regions.v1.interval_list

# Exome Project D5301, NA12892 Mother, Short reads
# BAM=/seq/picard_aggregation/D5301/NexPond-377874/current/NexPond-377874.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/exome_d5301_na12892_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/exome_d5301_na12892.vcf.gz
# INTERVAL=/seq/references/Homo_sapiens_assembly19/v1/variant_calling/exome_calling_regions.v1.interval_list

# Exome Project precisionFDA Hidden Treasure Challenge, NA12878
# BAM=/dsde/data/deep/vqsr/bams/NA12878MOD/NA12878MOD.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/exome_hidden_treasure_na12878_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/exome_hidden_treasure_na12878_v2.vcf.gz
#INTERVAL=/seq/references/Homo_sapiens_assembly19/v1/variant_calling/exome_calling_regions.v1.interval_list

# Exome Project D5227 NA12878
# BAM=/seq/picard_aggregation/D5227/NexPond-392292/current/NexPond-392292.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/exome_na12878_nexpond_392292.bam
# VCF=/dsde/data/deep/vqsr/vcfs/exome_na12878_nexpond_392292_all_calls.vcf.gz

# HG38 Project G94982, NA12878
# BAM=/dsde/data/deep/vqsr/bams/g94982_hg38.bam
# BAMOUT=/dsde/data/deep/vqsr/bams/g94982_hg38_na12878_bamout.bam
# VCF=/dsde/data/deep/vqsr/vcfs/g94982_hg38_na12878.vcf.gz
# REFERENCE=/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta
# INTERVAL=/seq/references/Homo_sapiens_assembly38/v0/variant_calling/wgs_calling_regions.v1.interval_list 


#Run Haplotype Caller
$ECHO \ 
$ECHO java -Djava.io.tmpdir=$TMPDIR -Dsamjdk.use_asysnc_io=true \
   -XX:GCTimeLimit=98 -XX:GCHeapFreeLimit=2 -Xmx55000m \
   -jar $QUEUE \
   -S $HAPLOTYPECALLER \
   -R $REFERENCE \
   -L $INTERVAL \
   -I $BAM \
   -bamout $BAMOUT \
   -dontTrimActive \
   -minConfidenceEmit 0 \
   -qsub -jobQueue gsa \
   -o $VCF \
   -sc $SCATTER \
   -memLimit 7 \
   -jobResReq virtual_free=8G \
   -mem 6


#Run VQSR and AssessNA12878
$ECHO \ 
$ECHO java -Djava.io.tmpdir=$TMPDIR -Dsamjdk.use_asysnc_io=true \
   -XX:GCTimeLimit=98 -XX:GCHeapFreeLimit=2 -Xmx16000m \
   -jar $QUEUE \
   -S $VQSR \
   -R $REFERENCE \
   -L $INTERVAL \
   --input_vcf $VCF \
   -an DP \
   -an QD \
   -an FS \
   -an SOR \
   -an MQ \
   -an MQRankSum \
   -an ReadPosRankSum \
   -oe _recalibrated.vcf.gz \
   -ie .vcf.gz \
   -indelMaxGaussians 5 \
   -qsub -jobQueue gsa


#Run Variant Annotator
$ECHO \ 
$ECHO java -jar $GATKUnstable \
   -R $REFERENCE \
   -T VariantAnnotator \
   -I $BAM \
   -o $VCF_annotated.vcf.gz  \
   -V $VCF \
   -L $VCF \
   --all


#Run Select Variants
$ECHO \ 
$ECHO java -jar $GATKUnstable \
   -T SelectVariants \
   -R $REFERENCE \
   -V /dsde/working/gauthier/ASHGposterResults/agilentParents_ICEoffspring_401trios_AS_VQSR.ICfiltered.recalibrated.vcf \
   -o agilent_parents_CHMI_CHMI3_Nex1.vcf \
   -sn CHMI_CHMI3_Nex1

$ECHO \ 
$ECHO java -jar $GATKUnstable \
   -T SelectVariants \
   -R $REFERENCE \
   -V /humgen/gsa-hpprojects/dev/gauthier/jointCallingTitration/parentsMax_oldRankSum_classicVQSR.ICfiltered.vcf.recalibrated.vcf \
   -env \
   -o joint_called_na12878.vcf \
   -sn NA12878




