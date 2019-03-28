#!/bin/bash
#test_table.sh
ECHO=echo
ARCHS="./weights/mix_hg19_rrab.json ./weights/g947_mix_1d_rab.json ./weights/nova_hiseq_mix_small.json ./weights/g947_site_labelled_rrac.json ./weights/g947_rr_c2.json ./weights/g947_reference_only.json ./weights/g947_mlp_th.json "
SAMPLES=3232

$ECHO python recipes.py test_architectures \
	--inspect_model \
	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--architectures $ARCHS \
	--id g94982_na12878_hg19_giab_test \
	--random_forest_training_sites ignore \
	--data_dir /dsde/data/deep/vqsr/tensors/g94982_na12878_hg19/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/nist_na12878_minimal.vcf.gz \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g94982_na12878.vcf.gz \
	--bed_file /dsde/data/deep/vqsr/beds/union13callableMQonlymerged_addcert_nouncert_excludesimplerep_excludesegdups_excludedecoy_excludeRepSeqSTRs_noCNVs_v2.18_2mindatasets_5minYesNoRatio.bed

$ECHO python recipes.py test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--architectures $ARCHS \
	--random_forest_training_sites ignore \
	--id g94982_na12878_hg19_platinum_test \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz \
	--data_dir /dsde/data/deep/vqsr/tensors/g94982_na12878_hg19_platinum/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/platinum_genomes/pg_hg19_na12878.vcf.gz \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g94982_na12878.vcf.gz \
	--bed_file /dsde/data/deep/vqsr/beds/platinum_genomes/pg_hg19_confident_regions.bed

$ECHO python recipes.py test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--architectures $ARCHS \
	--random_forest_training_sites ignore \
	--id g94982_na12878_hg19_pg_giab_hybrid_test \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/recalibrated_g94982.vcf.gz \
	--data_dir /dsde/data/deep/vqsr/tensors/g94982_na12878_hg19_pg_giab_hybrid/ \
	--bed_file /dsde/data/deep/vqsr/beds/platinum_genomes/pg_hg19_giab_hybrid.bed \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g94982_na12878.vcf.gz \
	--train_vcf /dsde/data/deep/vqsr/vcfs/platinum_genomes/pg_hg19_giab_hybrid_na12878.vcf.gz

$ECHO python recipes.py test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--architectures $ARCHS \
	--id g71602_na12877_hg19_test \
	--random_forest_training_sites ignore \
	--data_dir /dsde/data/deep/vqsr/tensors/g71602_hg19_na12877_platinum2/ \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/g71602_na12877_recalibrated.vcf.gz \
	--train_vcf /dsde/data/deep/vqsr/vcfs/platinum_genomes/pg_hg19_na12877.vcf.gz \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g71602_na12877.vcf.gz \
	--bed_file /dsde/data/deep/vqsr/beds/platinum_genomes/pg_hg19_confident_regions.bed

$ECHO python recipes.py test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--architectures $ARCHS \
	--id g94982_na12878_hg38_test \
	--data_dir /dsde/data/deep/vqsr/tensors/g94982_hg38_na12878/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/g947m_o1d1v1_na12878_recalibrated.vcf.gz \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g947m_o1d1v1_na12878.vcf.gz \
	--reference_fasta /seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta \
	--bed_file /dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed

$ECHO python recipes.py test_architectures \
 	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--architectures $ARCHS \
	--id g9486_na24385_hg38_test \
	--data_dir /dsde/data/deep/vqsr/tensors/g9486_na24385_hg38/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/giab_na24385_hg38_v3_3_2.vcf.gz \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/g947h_o1d1v1_na24385_recalibrated.vcf.gz \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g947h_o1d1v1_na24385.vcf.gz \
	--reference_fasta /seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta \
	--bed_file /dsde/data/deep/vqsr/beds/HG002_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC-SOLIDgatkHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed

$ECHO python recipes.py test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--architectures $ARCHS \
	--id g947p_na12878_hg38_test \
	--data_dir /dsde/data/deep/vqsr/tensors/g947p_na12878_hg38/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/g947m_o1d1v1_na12878_recalibrated.vcf.gz \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g947m_o1d1v1_na12878.vcf.gz \
	--reference_fasta /seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta \
	--bed_file /dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed


$ECHO python recipes.py test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--architectures $ARCHS \
	--sample_name SM-G947U \
	--id g947u_na24149_hg38_test \
	--data_dir /dsde/data/deep/vqsr/tensors/g947u_na24149/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/giab_na24149_hg38_v3_3_2.vcf.gz \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g947u_o1d1v2_na24149.vcf.gz \
	--reference_fasta /seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/g947u_o1d1v2_na24149_hc4_merged_recalibrated.vcf.gz \
	--bed_file //dsde/data/deep/vqsr/beds/HG003_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed


$ECHO python recipes.py test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--architectures $ARCHS \
	--id g947m_na12878_hg38_test \
	--data_dir /dsde/data/deep/vqsr/tensors/g947m_na12878/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/nist_na12878_giab_hg38.vcf.gz \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/g947m_o1d1v1_na12878_recalibrated.vcf.gz \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g947m_o1d1v1_na12878.vcf.gz \
	--reference_fasta /seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta \
	--bed_file /dsde/data/deep/vqsr/beds/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.bed


$ECHO python recipes.py test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--id syndip_hg38_test \
	--architectures $ARCHS \
	--train_vcf ../CHM-eval.kit/full.38.vcf.gz \
	--bed_file /dsde/data/deep/vqsr/beds/full.38.bed \
	--data_dir /dsde/data/deep/vqsr/tensors/g94794_wgs1_hg38_full_new/ \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g94794_chm_wgs1_hg38.vcf.gz \
	--reference_fasta /seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/g94794_chm_wgs1_hg38_hc4_merged_recalibrated.vcf.gz


$ECHO python recipes.py test_architectures \
	--gnomad_compare \
	--samples $SAMPLES \
	--single_sample_vqsr \
	--architectures $ARCHS \
	--id g947h_na24385_test \
	--data_dir /dsde/data/deep/vqsr/tensors/g947h_na24385_new/ \
	--train_vcf /dsde/data/deep/vqsr/vcfs/giab_na24385_hg38_v3_3_2.vcf.gz \
	--negative_vcf /dsde/data/deep/vqsr/vcfs/g947h_o1d1v1_na24385_recalibrated.vcf.gz \
	--deep_variant_vcf /dsde/data/deep/vqsr/vcfs/deep_variant_g947h_o1d1v1_na24385.vcf.gz \
	--reference_fasta /seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta \
	--bed_file /dsde/data/deep/vqsr/beds/HG002_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC-SOLIDgatkHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed


# python recipes.py test_architectures \
# 	--data_dir /tensors/g947h_na24385_channels_last/ \
# 	--architectures ./weights/nova_hiseq_mix_small_tf.json \
# 	--reference_fasta /ref/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta \
# 	--id g947h_na24385_test \
# 	--samples 1000 \
# 	--deep_variant_vcf /vcfs/deep_variant_g947h_o1d1v1_na24385.vcf.gz \
# 	--train_vcf /vcfs/giab_na24385_hg38_v3_3_2.vcf.gz \
# 	--negative_vcf /vcfs/g947h_o1d1v1_na24385_recalibrated.vcf.gz \
# 	--bed_file /beds/HG002_GRCh38_GIAB_highconf_CG-Illfb-IllsentieonHC-Ion-10XsentieonHC-SOLIDgatkHC_CHROM1-22_v.3.3.2_highconf_noinconsistent.bed 
