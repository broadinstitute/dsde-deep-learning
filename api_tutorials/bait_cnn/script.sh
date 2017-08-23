grep -v '^@' whole_exome_illumina_coding_v1.Homo_sapiens_assembly19.targets.interval_list | awk 'BEGIN{FS="\t";OFS="\t"}{print $1, $2-1,$3}' > whole_exome_illumina_coding_v1.Homo_sapiens_assembly19.targets.bed
grep -v '^@' whole_exome_illumina_coding_v1.Homo_sapiens_assembly19.baits.interval_list | awk 'BEGIN{FS="\t";OFS="\t"}{print $1, $2-1,$3}' > whole_exome_illumina_coding_v1.Homo_sapiens_assembly19.baits.bed

bedtools slop -i  whole_exome_illumina_coding_v1.Homo_sapiens_assembly19.baits.bed -b 250 -g hg19.genome > sloppy.baits.bed
bedtools coverage -counts -a whole_exome_illumina_coding_v1.Homo_sapiens_assembly19.targets.bed -b sloppy.baits.bed | awk '$4==1' > singleton.targets.bed
bedtools intersect  -a whole_exome_illumina_coding_v1.Homo_sapiens_assembly19.baits.bed -b singleton.targets.bed > singleton.baits.bed

tail -n+2 NexPond-647561.per_target_coverage | awk 'BEGIN{FS="\t";OFS="\t"}{print $1, $2-1,$3,$6,$14}' > coverage.bed
tail -n+2 NexPond-647561.coverage | awk 'BEGIN{FS="\t";OFS="\t"}{print $1, $2-1,$3,$6,$7}' > coverage2.bed
bedtools intersect -a coverage.bed -b singleton.baits.bed > coverage.singleton.bait.bed
bedtools intersect -a coverage2.bed -b singleton.baits.bed > temp
bedtools sort -i temp > coverage2.singleton.bait.bed 
