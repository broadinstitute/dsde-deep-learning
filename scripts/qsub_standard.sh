#! /bin/bash
# The Broad Institute
# SOFTWARE COPYRIGHT NOTICE AGREEMENT
# This software and its documentation are copyright 2009 by the
# Broad Institute/Massachusetts Institute of Technology. All rights are
# reserved.

# This software is supplied without any warranty or guaranteed support
# whatsoever. Neither the Broad Institute nor MIT can be responsible for its
# use, misuse, or functionality.

PROGNAME=`basename $0`
function usage () {
    echo "USAGE: $PROGNAME [options] -- program args..." >&2
    echo "-o <output_file>" >&2
    echo "-n <job_name>" >&2
    echo "-m <memory>" >&2
    echo "-s <num_slots>" >&2
    echo "-H <hold_for_jid> " >&2
    echo "[-x] (Do not send email)" >&2
    echo "offending argument: "$1
}


set -e
no_email=0
extra_qsub_args=
while getopts ":o:n:s:m:H:h:x\?" options; do
  case $options in
    o ) extra_qsub_args="$extra_qsub_args -o $OPTARG";;
    n ) extra_qsub_args="$extra_qsub_args -N $OPTARG";;
    m ) extra_qsub_args="$extra_qsub_args -l virtual_free=$OPTARG -l h_vmem=$OPTARG";;
    s ) extra_qsub_args="$extra_qsub_args -pe smp_pe $OPTARG";;
    H ) extra_qsub_args="$extra_qsub_args -hold_jid $OPTARG";;
    x ) no_email=1;;
    h ) usage $options
          exit 1;;
    \? ) usage $options
         exit 1;;
    * ) usage $options
          exit 1;;

  esac
done
shift $(($OPTIND - 1))


# -w e: reject invalid job
# -m e: send mail at end of job
# -M: email address to send to
# -j y: merge error and output streams
# -b y: treat command as binary
# -V: use environment of caller
# -cwd: set process wd to be cwd
if [ $no_email == 1 ];then
  qsub -w e -j y -b y -V -cwd -r y $extra_qsub_args "$@";
else
  qsub -w e -m e -M $(whoami)@broadinstitute.org -j y -b y -V -cwd -r y $extra_qsub_args "$@"
fi
