# variables
scriptfile=$1
nnodes=1
ncpus=24
queue=qexp
hours=01
min=00




echo -------------------------------------------------
echo -----------  run PyFETI scalability -------------
echo -------------------------------------------------
echo author: Guilherme Jenovencio
echo Running script = $1
qsub -A  $project_id -q $queue -l select=$nnodes:ncpus=$ncpus,walltime=$hours:$min:00 $scriptfile
