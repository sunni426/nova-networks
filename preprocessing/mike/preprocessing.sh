REPODIR="/dlab/ldrive/CBT/USLJ-DSDE-I10007/DSDE/shihch3/code/p/python/nova-networks"
DATADIR="/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/BenchmarkDatasets/hpa-single-cell-image-classification"
OPDATADIR="/dlab/ldrive/CBT/USLJ-DSDE_DATA-I10008/shihch3/projects/HPA_single_data/code_test"
CELLPOSE_CONDAENV="mscp03"

SCRIPTDIR="${REPODIR}/preprocessing/mike"
IPDIR="${DATADIR}/train"
NCPU=10

INPUTTYPE="train valid"
for type in $INPUTTYPE; do

    OPIMGDIR="${OPDATADIR}/${type}/img"
    OPIMGRGBDIR="${OPDATADIR}/${type}/imgRGB"
    OPSEGDIR="${OPDATADIR}/${type}/seg"
    OPCELLDIR="${OPDATADIR}/${type}/cell"
    CSVPATH="${REPODIR}/HPA-nova/dataloaders/split/${type}_sunni_toy.csv"
    FRAMESIZE=1024 # examples: "1024", "1024 1024", or "1024 512"

    # export 4ch png 
    python "${SCRIPTDIR}/channel_merge.py" \
        -i "${IPDIR}" \
        -ic "${CSVPATH}" \
        -o "${OPIMGDIR}" \
        -n $NCPU \
        -s $FRAMESIZE

    # export 3ch RGB png
    python "${SCRIPTDIR}/channel_merge.py" \
        -i "${IPDIR}" \
        -ic "${CSVPATH}" \
        -o "${OPIMGRGBDIR}" \
        -n $NCPU \
        -s $FRAMESIZE \
        -m # merge the red and yellow

    # create segmentation using cellpose
    # run with another conda env
    source activate ${CELLPOSE_CONDAENV} 
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
        # --max_cell_count 20 # <- default is 10 
done

# ========================================================================================

# use "all" for --max_cell_count can output all single cells. 
# Here is the example:  
OPCELLALLDIR="${OPDATADIR}/val/cell_all"
python "${SCRIPTDIR}/crop_img.py" \
    -ic "${CSVPATH}" \
    -ii "${OPIMGDIR}" \
    -is "${OPSEGDIR}" \
    -o "${OPCELLALLDIR}" \
    -n $NCPU \
    --max_cell_count all