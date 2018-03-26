/*
* By downloading the PROGRAM you agree to the following terms of use:
* 
* BROAD INSTITUTE
* SOFTWARE LICENSE AGREEMENT
* FOR ACADEMIC NON-COMMERCIAL RESEARCH PURPOSES ONLY
* 
* This Agreement is made between the Broad Institute, Inc. with a principal address at 415 Main Street, Cambridge, MA 02142 (“BROAD”) and the LICENSEE and is effective at the date the downloading is completed (“EFFECTIVE DATE”).
* 
* WHEREAS, LICENSEE desires to license the PROGRAM, as defined hereinafter, and BROAD wishes to have this PROGRAM utilized in the public interest, subject only to the royalty-free, nonexclusive, nontransferable license rights of the United States Government pursuant to 48 CFR 52.227-14; and
* WHEREAS, LICENSEE desires to license the PROGRAM and BROAD desires to grant a license on the following terms and conditions.
* NOW, THEREFORE, in consideration of the promises and covenants made herein, the parties hereto agree as follows:
* 
* 1. DEFINITIONS
* 1.1 PROGRAM shall mean copyright in the object code and source code known as GATK3 and related documentation, if any, as they exist on the EFFECTIVE DATE and can be downloaded from http://www.broadinstitute.org/gatk on the EFFECTIVE DATE.
* 
* 2. LICENSE
* 2.1 Grant. Subject to the terms of this Agreement, BROAD hereby grants to LICENSEE, solely for academic non-commercial research purposes, a non-exclusive, non-transferable license to: (a) download, execute and display the PROGRAM and (b) create bug fixes and modify the PROGRAM. LICENSEE hereby automatically grants to BROAD a non-exclusive, royalty-free, irrevocable license to any LICENSEE bug fixes or modifications to the PROGRAM with unlimited rights to sublicense and/or distribute.  LICENSEE agrees to provide any such modifications and bug fixes to BROAD promptly upon their creation.
* The LICENSEE may apply the PROGRAM in a pipeline to data owned by users other than the LICENSEE and provide these users the results of the PROGRAM provided LICENSEE does so for academic non-commercial purposes only. For clarification purposes, academic sponsored research is not a commercial use under the terms of this Agreement.
* 2.2 No Sublicensing or Additional Rights. LICENSEE shall not sublicense or distribute the PROGRAM, in whole or in part, without prior written permission from BROAD. LICENSEE shall ensure that all of its users agree to the terms of this Agreement. LICENSEE further agrees that it shall not put the PROGRAM on a network, server, or other similar technology that may be accessed by anyone other than the LICENSEE and its employees and users who have agreed to the terms of this agreement.
* 2.3 License Limitations. Nothing in this Agreement shall be construed to confer any rights upon LICENSEE by implication, estoppel, or otherwise to any computer software, trademark, intellectual property, or patent rights of BROAD, or of any other entity, except as expressly granted herein. LICENSEE agrees that the PROGRAM, in whole or part, shall not be used for any commercial purpose, including without limitation, as the basis of a commercial software or hardware product or to provide services. LICENSEE further agrees that the PROGRAM shall not be copied or otherwise adapted in order to circumvent the need for obtaining a license for use of the PROGRAM.
* 
* 3. PHONE-HOME FEATURE
* LICENSEE expressly acknowledges that the PROGRAM contains an embedded automatic reporting system (“PHONE-HOME”) which is enabled by default upon download. Unless LICENSEE requests disablement of PHONE-HOME, LICENSEE agrees that BROAD may collect limited information transmitted by PHONE-HOME regarding LICENSEE and its use of the PROGRAM.  Such information shall include LICENSEE’S user identification, version number of the PROGRAM and tools being run, mode of analysis employed, and any error reports generated during run-time.  Collection of such information is used by BROAD solely to monitor usage rates, fulfill reporting requirements to BROAD funding agencies, drive improvements to the PROGRAM, and facilitate adjustments to PROGRAM-related documentation.
* 
* 4. OWNERSHIP OF INTELLECTUAL PROPERTY
* LICENSEE acknowledges that title to the PROGRAM shall remain with BROAD. The PROGRAM is marked with the following BROAD copyright notice and notice of attribution to contributors. LICENSEE shall retain such notice on all copies. LICENSEE agrees to include appropriate attribution if any results obtained from use of the PROGRAM are included in any publication.
* Copyright 2012-2014 Broad Institute, Inc.
* Notice of attribution: The GATK3 program was made available through the generosity of Medical and Population Genetics program at the Broad Institute, Inc.
* LICENSEE shall not use any trademark or trade name of BROAD, or any variation, adaptation, or abbreviation, of such marks or trade names, or any names of officers, faculty, students, employees, or agents of BROAD except as states above for attribution purposes.
* 
* 5. INDEMNIFICATION
* LICENSEE shall indemnify, defend, and hold harmless BROAD, and their respective officers, faculty, students, employees, associated investigators and agents, and their respective successors, heirs and assigns, (Indemnitees), against any liability, damage, loss, or expense (including reasonable attorneys fees and expenses) incurred by or imposed upon any of the Indemnitees in connection with any claims, suits, actions, demands or judgments arising out of any theory of liability (including, without limitation, actions in the form of tort, warranty, or strict liability and regardless of whether such action has any factual basis) pursuant to any right or license granted under this Agreement.
* 
* 6. NO REPRESENTATIONS OR WARRANTIES
* THE PROGRAM IS DELIVERED AS IS. BROAD MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THE PROGRAM OR THE COPYRIGHT, EXPRESS OR IMPLIED, INCLUDING, WITHOUT LIMITATION, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NONINFRINGEMENT, OR THE ABSENCE OF LATENT OR OTHER DEFECTS, WHETHER OR NOT DISCOVERABLE. BROAD EXTENDS NO WARRANTIES OF ANY KIND AS TO PROGRAM CONFORMITY WITH WHATEVER USER MANUALS OR OTHER LITERATURE MAY BE ISSUED FROM TIME TO TIME.
* IN NO EVENT SHALL BROAD OR ITS RESPECTIVE DIRECTORS, OFFICERS, EMPLOYEES, AFFILIATED INVESTIGATORS AND AFFILIATES BE LIABLE FOR INCIDENTAL OR CONSEQUENTIAL DAMAGES OF ANY KIND, INCLUDING, WITHOUT LIMITATION, ECONOMIC DAMAGES OR INJURY TO PROPERTY AND LOST PROFITS, REGARDLESS OF WHETHER BROAD SHALL BE ADVISED, SHALL HAVE OTHER REASON TO KNOW, OR IN FACT SHALL KNOW OF THE POSSIBILITY OF THE FOREGOING.
* 
* 7. ASSIGNMENT
* This Agreement is personal to LICENSEE and any rights or obligations assigned by LICENSEE without the prior written consent of BROAD shall be null and void.
* 
* 8. MISCELLANEOUS
* 8.1 Export Control. LICENSEE gives assurance that it will comply with all United States export control laws and regulations controlling the export of the PROGRAM, including, without limitation, all Export Administration Regulations of the United States Department of Commerce. Among other things, these laws and regulations prohibit, or require a license for, the export of certain types of software to specified countries.
* 8.2 Termination. LICENSEE shall have the right to terminate this Agreement for any reason upon prior written notice to BROAD. If LICENSEE breaches any provision hereunder, and fails to cure such breach within thirty (30) days, BROAD may terminate this Agreement immediately. Upon termination, LICENSEE shall provide BROAD with written assurance that the original and all copies of the PROGRAM have been destroyed, except that, upon prior written authorization from BROAD, LICENSEE may retain a copy for archive purposes.
* 8.3 Survival. The following provisions shall survive the expiration or termination of this Agreement: Articles 1, 3, 4, 5 and Sections 2.2, 2.3, 7.3, and 7.4.
* 8.4 Notice. Any notices under this Agreement shall be in writing, shall specifically refer to this Agreement, and shall be sent by hand, recognized national overnight courier, confirmed facsimile transmission, confirmed electronic mail, or registered or certified mail, postage prepaid, return receipt requested. All notices under this Agreement shall be deemed effective upon receipt.
* 8.5 Amendment and Waiver; Entire Agreement. This Agreement may be amended, supplemented, or otherwise modified only by means of a written instrument signed by all parties. Any waiver of any rights or failure to act in a specific instance shall relate only to such instance and shall not be construed as an agreement to waive any rights or fail to act in any other instance, whether or not similar. This Agreement constitutes the entire agreement among the parties with respect to its subject matter and supersedes prior agreements or understandings between the parties relating to its subject matter.
* 8.6 Binding Effect; Headings. This Agreement shall be binding upon and inure to the benefit of the parties and their respective permitted successors and assigns. All headings are for convenience only and shall not affect the meaning of any provision of this Agreement.
* 8.7 Governing Law. This Agreement shall be construed, governed, interpreted and applied in accordance with the internal laws of the Commonwealth of Massachusetts, U.S.A., without regard to conflict of laws principles.
*/

package org.broadinstitute.gatk.queue.qscripts.techdev

import org.broadinstitute.gatk.queue.QScript
import org.broadinstitute.gatk.queue.extensions.gatk._

class VQSRScript extends QScript {
  val hapmap_sites = "/humgen/gsa-hpprojects/GATK/bundle/current/b37/hapmap_3.3.b37.vcf"
  val omni_1kg_sites = "/humgen/gsa-hpprojects/GATK/bundle/current/b37/1000G_omni2.5.b37.vcf"
  val high_conf_1kg_snps = "/humgen/gsa-hpprojects/GATK/bundle/current/b37/1000G_phase1.snps.high_confidence.b37.vcf"
  val dbsnp_sites = "/humgen/gsa-hpprojects/GATK/bundle/current/b37/dbsnp_138.b37.vcf"
  val mills_sites = "/humgen/gsa-hpprojects/GATK/bundle/current/b37/Mills_and_1000G_gold_standard.indels.b37.vcf"
  var snp_hapmap_resource = new TaggedFile(hapmap_sites, "hapmap,known=false,training=true,truth=true,prior=15.0")
  var snp_omni_resource = new TaggedFile(omni_1kg_sites, "omni,known=false,training=true,truth=true,prior=12.0")
  var snp_1kg_resource = new TaggedFile(high_conf_1kg_snps, "1000G,known=false,training=true,truth=false,prior=10.0")
  var dbsnp_resource = new TaggedFile(dbsnp_sites, "dbsnp,known=true,training=false,truth=false,prior=2.0")
  var indel_mills_resource = new TaggedFile(mills_sites, "mills,known=false,training=true,truth=true,prior=12.0")

  // Required arguments
  @Argument(shortName = "input", required = true, doc = "vcf file(s)") var input_vcf: File = _
  @Argument(shortName = "R", required = false, doc = "Reference sequence") var referenceFile: File = new File("/humgen/1kg/reference/human_g1k_v37_decoy.fasta")
  @Argument(shortName ="ts", required = false, doc = "ts filter level") var ts_filt_lev: Double = 99.9
  @Argument(shortName ="an", required = true, doc = "Annotations for VariantRcalibrator") var annot: List[String] = _
  @Argument(shortName = "sr", required = false, doc = "Snp resources for training") var snp_resource_list: List[TaggedFile] = List(snp_hapmap_resource, snp_omni_resource, snp_1kg_resource, dbsnp_resource)
  @Argument(shortName = "ir", required = false, doc = "Indel resources for training") var indel_resource_list: List[TaggedFile] = List(indel_mills_resource, dbsnp_resource)
  @Argument(shortName = "oe", required = false, doc = "output file extension") var output_extension = ".recalibrated.vcf"
  @Argument(shortName = "ie", required = false, doc = "input file extension") var input_extension = ".vcf"
  @Argument(shortName = "indelMaxGaussians", required = false, doc = "Indel max gaussians") var indelMaxGaussians: Int = 5

  // Script arguments
  @Argument(shortName = "sc", required = false, doc = "Scatter count") var jobs: Int = 1
  @Argument(shortName = "mem", required = false, doc = "memory for the jvm") var memoryLimit = 4
  @Argument(shortName = "assess", required = false, doc = "run NA12878 assessment") var assess: Boolean = false

  //
  // Walker optional arguments
  @Argument(shortName = "L", required = false, doc = "Intervals file") var intervalsFile: Seq[File] = _

  def script() {
    val snp_recal_file: File = swapExt("VQSR_tmp", input_vcf.getName, input_extension, ".snp.recal")
    val snp_tranches_file: File = swapExt("VQSR_tmp", input_vcf.getName, input_extension, ".snp.tranches")
    val indel_recal_file: File = swapExt("VQSR_tmp", input_vcf.getName, input_extension, ".indel.recal")
    val indel_tranches_file: File = swapExt("VQSR_tmp", input_vcf.getName, input_extension, ".indel.tranches")
    val snp_recalibrated_vcf: File = swapExt(input_vcf.getName, input_extension, ".raw_indels.snp_recalibrated.vcf")
    val snp_indel_recalibrated_vcf: File = swapExt(input_vcf.getName, input_extension, output_extension)

    val vr_snp = new VariantRecalibrator()
    vr_snp.intervals = intervalsFile
    vr_snp.reference_sequence = referenceFile
    vr_snp.input :+= input_vcf
    vr_snp.resource = snp_resource_list
    vr_snp.tranches_file = "results/" + snp_tranches_file
    vr_snp.recal_file = "results/" + snp_recal_file
    vr_snp.use_annotation ++= List("FS", "MQ", "QD", "DP", "SOR", "ReadPosRankSum", "MQRankSum")
    //vr_snp.use_annotation ++= List("FS", "QD", "SOR", "ReadPosRankSum", "MQRankSum")
    vr_snp.mode = org.broadinstitute.gatk.tools.walkers.variantrecalibration.VariantRecalibratorArgumentCollection.Mode.SNP

    // Script arguments
    vr_snp.memoryLimit = memoryLimit

    // Walker optional arguments
    val ar_snp = new ApplyRecalibration()
    ar_snp.intervals = intervalsFile
    ar_snp.reference_sequence = referenceFile
    ar_snp.input :+= input_vcf
    ar_snp.tranches_file = "results/" + snp_tranches_file
    ar_snp.recal_file = "results/" + snp_recal_file
    ar_snp.ts_filter_level = ts_filt_lev
    ar_snp.mode = org.broadinstitute.gatk.tools.walkers.variantrecalibration.VariantRecalibratorArgumentCollection.Mode.SNP
    ar_snp.out = "results/" + snp_recalibrated_vcf

    // Script arguments
    ar_snp.scatterCount = jobs
    ar_snp.memoryLimit = memoryLimit

    val vr_indel = new VariantRecalibrator()
    vr_indel.intervals = intervalsFile
    vr_indel.reference_sequence = referenceFile
    vr_indel.input :+= "results/" + snp_recalibrated_vcf
    vr_indel.resource = indel_resource_list
    vr_indel.tranches_file = "results/" + indel_tranches_file
    vr_indel.recal_file = "results/" + indel_recal_file
    vr_indel.use_annotation ++= List("FS", "QD", "SOR", "ReadPosRankSum", "MQRankSum")
    vr_indel.mode = org.broadinstitute.gatk.tools.walkers.variantrecalibration.VariantRecalibratorArgumentCollection.Mode.INDEL
    vr_indel.maxGaussians = indelMaxGaussians

    // Script arguments
    vr_indel.memoryLimit = memoryLimit

    val ar_indel = new ApplyRecalibration()
    ar_indel.intervals = intervalsFile
    ar_indel.reference_sequence = referenceFile
    ar_indel.input :+= "results/" + snp_recalibrated_vcf
    ar_indel.tranches_file = "results/" + indel_tranches_file
    ar_indel.recal_file = "results/" + indel_recal_file
    ar_indel.ts_filter_level = ts_filt_lev
    ar_indel.mode = org.broadinstitute.gatk.tools.walkers.variantrecalibration.VariantRecalibratorArgumentCollection.Mode.INDEL
    ar_indel.out = "results/" + snp_indel_recalibrated_vcf

    // Script arguments
    ar_indel.scatterCount = jobs
    ar_indel.memoryLimit = memoryLimit

  	if (assess) {
    	val kb = new AssessNA12878
    	kb.variant :+= "results/" + swapExt(input_vcf, input_extension, ".recalibrated.vcf")
    	kb.intervals = intervalsFile
    	kb.reference_sequence = referenceFile
    	kb.memoryLimit = memoryLimit
    	kb.badSites = "results/" + swapExt(input_vcf, input_extension, ".bad")
    	kb.allSites = true
    	kb.detailed = false
    	add(kb)
    }
    
    add(vr_snp)
    add(ar_snp)
    add(vr_indel)
    add(ar_indel)
  }
}

