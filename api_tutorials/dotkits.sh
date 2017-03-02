. /broad/software/scripts/useuse
use Anaconda
use .cuda-7.0-ea
use .gcc-4.9.1
use .openblas-0.2.8

export LD_LIBRARY_PATH=/dsde/working/sam/cnn/cuda/lib64/:$LD_LIBRARY_PATH
export CPATH=/dsde/working/sam/cnn/cuda/include/:$CPATH
export LIBRARY_PATH=/dsde/working/sam/cnn/cuda/lib64:$LIBRARY_PATH
