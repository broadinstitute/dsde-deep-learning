# CNNScoreVariants WDL

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
    Int default_ram_mb = 3000
    # WARNING: In the workflow, you should calculate the disk space as an input to this task (disk_space_gb).  Please see [TODO: Link from Jose] for examples.
    Int default_disk_space_gb = 100

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem_gb) then mem_gb *1000 else default_ram_mb
    Int command_mem = machine_mem - 1000

command <<<
    
        set -e
        export GATK_LOCAL_JAR=${default="/root/gatk.jar" gatk4_jar_override}

        #gatk --java-options "-Xmx${command_mem}m" \
        java "-Xmx${command_mem}m" -Djava.io.tmpdir=tmp -jar ${gatk4_jar_override} \
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

task MergeVCFs {
    Array[File] input_vcfs
    String output_vcf_name   
    File? gatk4_jar_override

    # Runtime parameters
    Int? mem_gb
    String gatk_docker
    Int? preemptible_attempts
    Int? disk_space_gb
    Int? cpu 
    Boolean use_ssd = false

    # You may have to change the following two parameter values depending on the task requirements
    Int default_ram_mb = 3000
    # WARNING: In the workflow, you should calculate the disk space as an input to this task (disk_space_gb).  Please see [TODO: Link from Jose] for examples.
    Int default_disk_space_gb = 100

    # Mem is in units of GB but our command and memory runtime values are in MB
    Int machine_mem = if defined(mem_gb) then mem_gb *1000 else default_ram_mb
    Int command_mem = machine_mem - 1000

command <<<   
        set -e
        export GATK_LOCAL_JAR=${default="/root/gatk.jar" gatk4_jar_override}
        #gatk --java-options "-Xmx${command_mem}m" \
        java -Xmx2g -Djava.io.tmpdir=tmp -jar ${gatk4_jar_override} \
            MergeVcfs -I ${sep=' -I ' input_vcfs} \
            -O "${output_vcf_name}"
>>>
  runtime {
    docker: gatk_docker
    memory: machine_mem + " MB"
    # Note that the space before SSD and HDD should be included.
    disks: "local-disk " + select_first([disk_space_gb, default_disk_space_gb]) + if use_ssd then " SSD" else " HDD"
    preemptible: select_first([preemptible_attempts, 3])
    cpu: select_first([cpu, 1])  
  }
  output {
    File output_vcf = "${output_vcf_name}"
    File output_vcf_index = "${output_vcf_name}.tbi"
  }
}

workflow CNNScoreVariantsWorkflow {
    File input_vcf
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
    File gatk4_jar_override
    String gatk_docker
    File picard_jar
    File calling_intervals
    Int scatter_count 
    String final_output_name
    Int? preemptible_attempts
    Int? mem_gb

    call SplitIntervals {
        input:
            picard_jar = picard_jar,
            scatter_count = scatter_count,
            intervals = calling_intervals
    }

    scatter (calling_interval in SplitIntervals.interval_files) {

        call CNNScoreVariants {
            input:
                input_vcf = input_vcf,
                input_vcf_index = input_vcf_index,
                reference_fasta = reference_fasta,
                reference_dict = reference_dict,
                reference_fasta_index = reference_fasta_index,
                bam_file = bam_file,
                bam_file_index = bam_file_index,
                architecture_json = architecture_json,
                architecture_hd5 = architecture_hd5,
                tensor_type = tensor_type,
                inference_batch_size = inference_batch_size,
                transfer_batch_size = transfer_batch_size,
                output_prefix = output_prefix,
                interval_list = calling_interval,
                gatk4_jar_override = gatk4_jar_override,
                gatk_docker = gatk_docker,
                preemptible_attempts = preemptible_attempts,
                mem_gb = mem_gb
        }
    }

    call MergeVCFs as MergeVCF_CNN {
        input: 
            input_vcfs = CNNScoreVariants.cnn_annotated_vcf,
            output_vcf_name = final_output_name,
            gatk4_jar_override = gatk4_jar_override,
            gatk_docker = gatk_docker
    }

    output {
        MergeVCF_CNN.*
    }
}