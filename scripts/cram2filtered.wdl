# This workflow takes a CRAM calls variants with HaplotypeCaller and filters the calls with a Neural Net
workflow Cram2FilteredVcf {
    File input_cram # Aligned CRAM file
    File reference_fasta 
    File reference_dict
    File reference_fasta_index
    File architecture_json # Neural Net configuration for CNNScoreVariants  
    File architecture_hd5  # Pre-Trained weights and architecture for CNNScoreVariants
    Int inference_batch_size # Batch size for python in CNNScoreVariants  
    Int transfer_batch_size  # Batch size for java in CNNScoreVariants     
    String output_prefix # Identifying string for this run will be used to name all output files
    String tensor_type # What kind of tensors the Neural Net expects (e.g. reference, read_tensor)
    File? gatk4_jar_override
    String gatk_docker    
    File picard_jar  
    File calling_intervals
    Int scatter_count # Number of shards for parrallelization of HaplotypeCaller and CNNScoreVariants
    String extra_args # Extra arguments for HaplotypeCaller

    # Runtime parameters
    Int? mem_gb
    Int? preemptible_attempts
    Int? disk_space_gb
    Int? cpu 

    call CramToBam {
        input:
          reference_fasta = reference_fasta,
          reference_dict = reference_dict,
          reference_fasta_index = reference_fasta_index,
          cram_file = input_cram,
          output_prefix = output_prefix,
          disk_space_gb = disk_space_gb
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
                gatk_docker = gatk_docker,
                gatk4_jar_override = gatk4_jar_override,
                extra_args = extra_args,
                disk_space_gb = disk_space_gb
        }

        call CNNScoreVariants {
            input:
                input_vcf = RunHC4.raw_vcf,
                input_vcf_index = RunHC4.raw_vcf_index,
                bam_file = RunHC4.bamout,
                bam_file_index = RunHC4.bamout_index,
                architecture_json = architecture_json,
                architecture_hd5 = architecture_hd5,
                reference_fasta = reference_fasta,
                tensor_type = tensor_type,
                inference_batch_size = inference_batch_size,
                transfer_batch_size = transfer_batch_size,
                reference_dict = reference_dict,
                reference_fasta_index = reference_fasta_index,               
                output_prefix = output_prefix,
                interval_list = calling_interval,
                gatk4_jar_override = gatk4_jar_override,
                gatk_docker = gatk_docker,
                preemptible_attempts = preemptible_attempts,
                mem_gb = mem_gb
        }
    }

    call MergeVCFs as MergeVCF_HC4 {
        input: 
            input_vcfs = CNNScoreVariants.cnn_annotated_vcf,
            output_prefix = output_prefix,
            picard_jar = picard_jar
    }

    call SamtoolsMergeBAMs {
        input:
            input_bams = RunHC4.bamout,
            output_prefix = output_prefix,
            picard_jar = picard_jar
    }
  


    output {
        MergeVCF_HC4.*
        SamtoolsMergeBAMs.*
    }

}


task CramToBam {
    File reference_fasta
    File reference_fasta_index
    File reference_dict
    File cram_file
    String output_prefix

    # Runtime parameters
    Int? mem_gb
    Int? preemptible_attempts
    Int? disk_space_gb
    Int? cpu 
    Boolean use_ssd = false

    # You may have to change the following two parameter values depending on the task requirements
    Int default_ram_mb = 16000
    # WARNING: In the workflow, you should calculate the disk space as an input to this task (disk_space_gb).
    Int default_disk_space_gb = 200

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem_gb) then mem_gb *1000 else default_ram_mb
    Int command_mem = machine_mem - 1000

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
    memory: machine_mem + " MB"
    # Note that the space before SSD and HDD should be included.
    disks: "local-disk " + select_first([disk_space_gb, default_disk_space_gb]) + if use_ssd then " SSD" else " HDD"
    preemptible: select_first([preemptible_attempts, 3])
    cpu: select_first([cpu, 1])  
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
    String extra_args
    File? gatk4_jar_override

    # Runtime parameters
    Int? mem_gb
    String gatk_docker
    Int? preemptible_attempts
    Int? disk_space_gb
    Int? cpu 
    Boolean use_ssd = false

    # You may have to change the following two parameter values depending on the task requirements
    Int default_ram_mb = 8000
    # WARNING: In the workflow, you should calculate the disk space as an input to this task (disk_space_gb).  Please see [TODO: Link from Jose] for examples.
    Int default_disk_space_gb = 200

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem_gb) then mem_gb *1000 else default_ram_mb
    Int command_mem = machine_mem - 1000

    command {
        set -e
        export GATK_LOCAL_JAR=${default="/root/gatk.jar" gatk4_jar_override}

        gatk --java-options "-Xmx${command_mem}m" \        
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
        #docker: gatk_docker
        docker: "samfriedman/p3"
        memory: machine_mem + " MB"
        # Note that the space before SSD and HDD should be included.
        disks: "local-disk " + select_first([disk_space_gb, default_disk_space_gb]) + if use_ssd then " SSD" else " HDD"
        preemptible: select_first([preemptible_attempts, 3])
        cpu: select_first([cpu, 1])  
    }
}

task CNNScoreVariants {

    String input_vcf
    File input_vcf_index
    File reference_fasta
    File reference_dict
    File reference_fasta_index
    String? bam_file
    String? bam_file_index
    File architecture_json
    File architecture_hd5
    String tensor_type
    String output_prefix
    Int inference_batch_size
    Int transfer_batch_size
    File interval_list
    File? gatk4_jar_override

    # Runtime parameters
    Int? mem_gb
    String gatk_docker
    Int? preemptible_attempts
    Int? disk_space_gb
    Int? cpu 
    Boolean use_ssd = false

    String bam_cl = if defined(bam_file) then "-I ${bam_file}" else " "

    # You may have to change the following two parameter values depending on the task requirements
    Int default_ram_mb = 16000
    # WARNING: In the workflow, you should calculate the disk space as an input to this task (disk_space_gb).  Please see [TODO: Link from Jose] for examples.
    Int default_disk_space_gb = 200

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem_gb) then mem_gb *1000 else default_ram_mb
    Int command_mem = machine_mem - 1000

command <<<
        set -e
        export GATK_LOCAL_JAR=${default="/root/gatk.jar" gatk4_jar_override}

        gatk --java-options "-Xmx${command_mem}m" \
        CNNScoreVariants \
        ${bam_cl} \
        -R ${reference_fasta} \
        -V ${input_vcf} \
        -O ${output_prefix}_cnn_annotated.vcf.gz \
        -L ${interval_list} \
        --tensor-type ${tensor_type} \
        --inference-batch-size ${inference_batch_size} \
        --transfer-batch-size ${transfer_batch_size}  \
        --architecture ${architecture_json}
>>>

  runtime {
    #docker: gatk_docker
    docker: "samfriedman/p3"
    memory: machine_mem + " MB"
    # Note that the space before SSD and HDD should be included.
    disks: "local-disk " + select_first([disk_space_gb, default_disk_space_gb]) + if use_ssd then " SSD" else " HDD"
    preemptible: select_first([preemptible_attempts, 3])
    cpu: select_first([cpu, 1])  
  }

  output {
    File cnn_annotated_vcf = "${output_prefix}_cnn_annotated.vcf.gz"
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
