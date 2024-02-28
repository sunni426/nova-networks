SCRIPTDIR="/dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/nova-networks/preprocessing/mike"
DATADIR="/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/BenchmarkDatasets/hpa-single-cell-image-classification"
IPDIR="${DATADIR}/train"
NCPU=10

OPDATADIR="/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data/code_test"
OPIMGDIR="${OPDATADIR}/train/img"
OPIMGRGBDIR="${OPDATADIR}/train/imgRGB"
OPSEGDIR="${OPDATADIR}/train/seg"
OPCELLDIR="${OPDATADIR}/train/cell"
CSVPATH="/dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/nova-networks/HPA-nova/dataloaders/split/train_sunni_toy.csv"
FRAMESIZE=1024 # examples: "1024", "1024 1024", or "1024 512"

python "${SCRIPTDIR}/channel_merge.py" \
    -i "${IPDIR}" \
    -ic "${CSVPATH}" \
    -o "${OPIMGDIR}" \
    -n $NCPU \
    -s $FRAMESIZE

python "${SCRIPTDIR}/channel_merge.py" \
    -i "${IPDIR}" \
    -ic "${CSVPATH}" \
    -o "${OPIMGRGBDIR}" \
    -n $NCPU \
    -s $FRAMESIZE \
    -m

source activate mscp03
python "${SCRIPTDIR}/cellpose_seg.py" \
    -i "${OPIMGRGBDIR}" \
    -o "${OPSEGDIR}"
conda deactivate

python "${SCRIPTDIR}/crop_img.py" \
    -ic "${CSVPATH}" \
    -ii "${OPIMGDIR}" \
    -is "${OPSEGDIR}" \
    -o "${OPCELLDIR}" \
    -n $NCPU

OPIMGDIR="${OPDATADIR}/val/img"
OPIMGRGBDIR="${OPDATADIR}/val/imgRGB"
OPSEGDIR="${OPDATADIR}/val/seg"
OPCELLDIR="${OPDATADIR}/val/cell"
CSVPATH="/dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/nova-networks/HPA-nova/dataloaders/split/valid_sunni_toy.csv"
FRAMESIZE=1024 # examples: "1024", "1024 1024", or "1024 512"

python "${SCRIPTDIR}/channel_merge.py" \
    -i "${IPDIR}" \
    -ic "${CSVPATH}" \
    -o "${OPIMGDIR}" \
    -n $NCPU \
    -s $FRAMESIZE

python "${SCRIPTDIR}/channel_merge.py" \
    -i "${IPDIR}" \
    -ic "${CSVPATH}" \
    -o "${OPIMGRGBDIR}" \
    -n $NCPU \
    -s $FRAMESIZE \
    -m

source activate mscp03
python "${SCRIPTDIR}/cellpose_seg.py" \
    -i "${OPIMGRGBDIR}" \
    -o "${OPSEGDIR}"
conda deactivate

python "${SCRIPTDIR}/crop_img.py" \
    -ic "${CSVPATH}" \
    -ii "${OPIMGDIR}" \
    -is "${OPSEGDIR}" \
    -o "${OPCELLDIR}" \
    -n $NCPU