#!/bin/bash
ECHO=echo
SAMPLES=3200
ARCHS="./weights/pg_giab_hybridd_hg19_rra_small.json ./weights/pg_platinum_fix_rra_small.json "
ARCHS=$ARCHS" ./weights/g94982_hg19_giab_small_new.json ./weights/g94794_wgs1_hg38_small_new.json "
ARCHS=$ARCHS" ./weights/xna12878_exome_small_new.json ./weights/syndip_exome_small_new.json "
ARCHS=$ARCHS" ./weights/nova_g947m_small_new.json ./weights/nova_g947t_small_new.json"
ARCHS=$ARCHS" ./weights/g947h_small_new.json"
ARCHS=$ARCHS" ./weights/xxx_g947_small_2d.json ./weights/xxx_nova_hiseq_mix_small.json"
# ./weights/rra_b_exome_bp_th.json 

$ECHO python recipes.py \
	test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--architectures $ARCHS \
	--id syndip_hg38_gen_test \
	--baseline_key GATK4_g94794_wgs1_hg38_small_new \
	--bed_file /dsde/working/sam/CHM-eval.kit/full.38.bed \
	--train_vcf /dsde/working/sam/CHM-eval.kit/full.38.vcf.gz \
	--data_dir /dsde/data/deep/vqsr/tensors/g94794_wgs1_hg38_full_new/ \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g94794_chm_wgs1_hg38.vcf.gz \
	--reference_fasta /seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/g94794_chm_wgs1_hg38_hc4_merged_recalibrated.vcf.gz

$ECHO python recipes.py \
	test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--architectures $ARCHS \
	--baseline_key GATK4_g947h_small_new \
	--id g947h_na24385_hg38_GiaB_gen_test \
	--data_dir /dsde/data/deep/vqsr/tensors/g947h_na24385_new/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/giab_na24385_hg38_v3_3_2.vcf.gz \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/g947h_o1d1v1_na24385_hc4_merged.vcf.gz \
	--reference_fasta /seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta \
	--bed_file /dsde/data/deep/vqsr/beds/HG002_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC-SOLIDgatkHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

$ECHO python recipes.py \
	test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--architectures $ARCHS \
	--random_forest_training_sites ignore \
	--id g94982_hg19_giab_small_new_gen_test \
	--baseline_key GATK4_g94982_hg19_giab_small_new \
	--data_dir /dsde/data/deep/vqsr/tensors/g94982_na12878_hg19/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g94982_na12878.vcf.gz \
	--bed_file /dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

$ECHO python recipes.py \
	test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--architectures $ARCHS \
	--baseline_key GATK4_nova_g947m_small_new \
	--id nova_g947m_na12878_hg38_GiaB_gen_test \
	--data_dir /dsde/data/deep/vqsr/tensors/nova_g947m_na12878_hg38_new/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/nova_g947m_na12878_cnn_scored.vcf.gz \
	--reference_fasta /seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta \
	--bed_file /dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

$ECHO python recipes.py \
	test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--architectures $ARCHS \
	--baseline_key GATK4_nova_g947t_small_new \
	--id nova_g947t_na24385_hg38_GiaB_gen_test \
	--data_dir /dsde/data/deep/vqsr/tensors/nova_g947t_na24385_hg38/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/giab_na24385_hg38_v3_3_2.vcf.gz \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/nova_g947t_na24385_cnn_scored.vcf.gz \
	--reference_fasta /seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta \
	--bed_file /dsde/data/deep/vqsr/beds/HG002_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC-SOLIDgatkHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

$ECHO python recipes.py \
	test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--architectures $ARCHS \
	--random_forest_training_sites ignore \
	--id g94982_na12878_hg19_giab_gen_test \
	--baseline_key GATK4_pg_giab_hybridd_hg19_rra_small \
	--data_dir /dsde/data/deep/vqsr/tensors/g94982_na12878_hg19_pg_giab_hybrid/ \
	--bed_file /dsde/data/deep/vqsr/beds/platinum_genomes/pg_hg19_giab_hybrid.bed \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g94982_na12878.vcf.gz \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/na12878_g94982_bamout.raw.snps.indels.vcf.gz \
	--train_vcf /dsde/data/deep/vqsr/vcfs/platinum_genomes/pg_hg19_giab_hybrid_na12878.vcf.gz

$ECHO python recipes.py \
	test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--architectures $ARCHS \
	--random_forest_training_sites ignore \
	--id g94982_na12878_hg19_platinum_gen_test \
	--baseline_key GATK4_pg_platinum_fix_rra_small \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz \
	--data_dir /dsde/data/deep/vqsr/tensors/g94982_na12878_hg19_platinum_fix/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/platinum_genomes/pg_hg19_na12878.vcf.gz \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g94982_na12878.vcf.gz \
	--bed_file /dsde/data/deep/vqsr/beds/platinum_genomes/ConfidentRegions_pg_b37_2017.bed

$ECHO python recipes.py \
	test_architectures \
	--gnomad_compare  \
	--samples $SAMPLES \
	--architectures $ARCHS \
	--id exome_syndip_gen_test \
	--baseline_key GATK4_syndip_exome_small_new \
	--bed_file /dsde/working/sam/CHM-eval.kit/full.37m.bed \
	--train_vcf /dsde/working/sam/CHM-eval.kit/full.37m.vcf.gz \
	--data_dir  /dsde/data/deep/vqsr/tensors/exome_syndip_mix2/ \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_chm_wgs1.vcf.gz \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/exome_c1963_chmi_chmi3_nex1.vcf.gz 

$ECHO python recipes.py \
	test_architectures \
	--gnomad_compare  \
	--samples $SAMPLES \
	--architectures $ARCHS \
	--id exome_na12878_gen_test \
	--baseline_key GATK4_xna12878_exome_small_new \
	--data_dir  /dsde/data/deep/vqsr/tensors/exome_na12878_mix_hg19/ \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/exome_d5800_na12878.vcf.gz 

