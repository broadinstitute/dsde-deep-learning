# CRAM to filtered VCF WDL
task CramToBam {
  File reference_fasta
  File reference_fasta_index
  File reference_dict
  File cram_file
  String output_prefix

  Int disk_size

command <<<
  ls -ltr ${cram_file} ${reference_fasta} &&
  echo "ls (1): complete" &&
  samtools view -h -T ${reference_fasta} ${cram_file} |
  samtools view -b -o ${output_prefix}.bam - &&
  echo "samtools view: complete" &&
  ls -ltr . &&
  echo "ls (2): complete" &&
  samtools index -b ${output_prefix}.bam &&
  echo "samtools index: complete" &&
  ls -ltr . &&
  echo "ls (3): complete" &&
  mv ${output_prefix}.bam.bai ${output_prefix}.bai &&
  echo "mv: complete" &&
  ls -ltr . &&
  echo "ls (4): complete"
  >>>
  runtime {
    docker: "broadinstitute/genomes-in-the-cloud:2.1.1"
    memory: "3 GB"
    disks: "local-disk " + disk_size + " HDD"
  }
  output {
    File output_bam = "${output_prefix}.bam"
    File output_bam_index = "${output_prefix}.bai"
  }
}


task RunHC4 {
    File input_bam
    File input_bam_index
    File reference_fasta
    File reference_dict
    File reference_fasta_index
    String output_prefix
    File interval_list
    File gatk_jar
    String extra_args
    Int disk_size

    command { 
        java -Djava.io.tmpdir=tmp -jar ${gatk_jar} \
        HaplotypeCaller \
        -R ${reference_fasta} \
        -I ${input_bam} \
        -O ${output_prefix}_hc4.vcf.gz \
        -L ${interval_list} \
        -bamout ${output_prefix}_bamout.bam \
        ${extra_args}
    }

    output {
        File bamout = "${output_prefix}_bamout.bam"
        File bamout_index = "${output_prefix}_bamout.bai"
        File raw_vcf = "${output_prefix}_hc4.vcf.gz"
        File raw_vcf_index = "${output_prefix}_hc4.vcf.gz.tbi"
    }
    runtime {
        docker: "broadinstitute/genomes-in-the-cloud:2.1.1"
        memory: "3 GB"
        disks: "local-disk " + disk_size + " HDD"
    }
}

task WriteTensors {
    File input_bam
    File input_bam_index
    File input_vcf
    File input_vcf_index
    File reference_fasta
    File reference_dict
    File reference_fasta_index
    File truth_vcf
    File truth_vcf_index
    File truth_bed
    String output_prefix
    String tensor_dir
    File interval_list
    File gatk_jar
    Int disk_size

    command {
        java -Djava.io.tmpdir=tmp -jar ${gatk_jar} \
        CNNVariantWriteTensors \
        -R ${reference_fasta} \
        -V ${input_vcf} \
        -truth-vcf ${truth_vcf} \
        -truth-bed ${truth_bed} \
        -tensor-type read_tensor \
        -output-tensor-dir "./tensors/" \
        -bam-file ${input_bam}
        
        if [ -d "./tensors/" ]; then
            gsutil -q -m cp -R "./tensors/" ${tensor_dir}
        fi
    }

    output {
      #File tensors = "tensors.tar.gz"
      #Array[File] tensors=glob("./tensors/*/*/*.hd5")
    }
    runtime {
        docker: "samfriedman/p3"
        memory: "7 GB"
        disks: "local-disk " + disk_size + " HDD"
    }

}


task TrainModel {
    String tensor_dir
    String output_prefix
    File gatk_jar
    Int disk_size

    command { 
        java -Djava.io.tmpdir=tmp -jar ${gatk_jar} \
        CNNVariantTrain \
        -input-tensor-dir ${tensor_dir} \
        -model-name ${output_prefix} \
        -tensor-type read_tensor
    }

    output {

    }

    runtime {
      docker: "samfriedman/gpu"
      gpuType: "nvidia-tesla-k80" 
      gpuCount: 1 
      zones: ["us-central1-c"]
      memory: "16 GB"
      disks: "local-disk 400 HDD"
      bootDiskSizeGb: "16" 
    }
}


task MergeVCFs {
    Array[File] input_vcfs
    String output_prefix
    File picard_jar
    command {
        java -jar ${picard_jar} \
        MergeVcfs \
        INPUT=${sep=' INPUT=' input_vcfs} \
        OUTPUT=${output_prefix}_hc4_merged.vcf.gz
    }

    output {
        File output_vcf = "${output_prefix}_hc4_merged.vcf.gz"
    }

    runtime {
        docker: "broadinstitute/genomes-in-the-cloud:2.1.1"
        memory: "16 GB"
        disks: "local-disk 400 HDD"
    } 
}

task SamtoolsMergeBAMs {
    Array[File] input_bams
    String output_prefix
    File picard_jar
    command {
        samtools merge ${output_prefix}_bamout.bam ${sep=' ' input_bams} 
    }

    output {
        File bamout = "${output_prefix}_bamout.bam"
    }

  runtime {
    docker: "broadinstitute/genomes-in-the-cloud:2.1.1"
    memory: "16 GB"
    disks: "local-disk 400 HDD"
  }    
}

task SplitIntervals {
    File picard_jar
    Int scatter_count
    File intervals

    command {
        set -e
        Picard_Jar=${picard_jar}
        mkdir interval-files
        java -Xmx6g -jar $Picard_Jar IntervalListTools I=${intervals} O=interval-files SCATTER_COUNT=${scatter_count}
        find ./interval-files -iname scattered.interval_list | sort > interval-files.txt
        i=1
        while read file; 
        do
            mv $file $i.interval_list
            ((i++))
        done < interval-files.txt
    }
    
    output {
        Array[File] interval_files = glob("*.interval_list")
    }

    runtime {
         docker: "samfriedman/p3"
         memory: "3 GB"
         cpu: "1"
         disks: "local-disk 200 HDD"
    }
}

workflow HC4Workflow {
    File input_cram
    File reference_fasta
    File reference_dict
    File reference_fasta_index
    File truth_vcf
    File truth_vcf_index
    File truth_bed
    String output_prefix
    String tensor_dir
    File gatk4_jar
    File picard_jar
    File calling_intervals
    Int scatter_count
    Int disk_size
    String extra_args

    call CramToBam {
        input:
          reference_fasta = reference_fasta,
          reference_dict = reference_dict,
          reference_fasta_index = reference_fasta_index,
          cram_file = input_cram,
          output_prefix = output_prefix,
          disk_size = disk_size
    }

    call SplitIntervals {
        input:
            picard_jar = picard_jar,
            scatter_count = scatter_count,
            intervals = calling_intervals
    }

    scatter (calling_interval in SplitIntervals.interval_files) {

        call RunHC4 {
            input:
                input_bam = CramToBam.output_bam,
                input_bam_index = CramToBam.output_bam_index,
                reference_fasta = reference_fasta,
                reference_dict = reference_dict,
                reference_fasta_index = reference_fasta_index,
                output_prefix = output_prefix,
                interval_list = calling_interval,
                gatk_jar = gatk4_jar,
                extra_args = extra_args,
                disk_size = disk_size
                
        }

        call WriteTensors {
            input:
                input_vcf = RunHC4.raw_vcf,
                input_vcf_index = RunHC4.raw_vcf_index,
                input_bam = RunHC4.bamout,
                input_bam_index = RunHC4.bamout_index,
                truth_vcf = truth_vcf,
                truth_vcf_index = truth_vcf_index,
                truth_bed = truth_bed,
                reference_fasta = reference_fasta,
                reference_dict = reference_dict,
                reference_fasta_index = reference_fasta_index,               
                output_prefix = output_prefix,
                tensor_dir = tensor_dir,
                interval_list = calling_interval,
                gatk_jar = gatk4_jar,
                disk_size = disk_size
        }
    }

    call MergeVCFs as MergeVCF_HC4 {
        input: 
            input_vcfs = RunHC4.raw_vcf,
            output_prefix = output_prefix,
            picard_jar = picard_jar
    }

    call SamtoolsMergeBAMs {
        input:
            input_bams = RunHC4.bamout,
            output_prefix = output_prefix,
            picard_jar = picard_jar
    }

#    call TrainModel {
#        input:
#            tensor_dir = tensor_dir,
#            output_prefix = output_prefix,
#            gatk_jar = gatk4_jar,
#            disk_size = disk_size
#    }    


    output {
        MergeVCF_HC4.*
        SamtoolsMergeBAMs.*
        #TrainModel.*
    }

}