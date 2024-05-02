

#=================================================================================================================================

REPODIR="/projectnb/btec-design3/novanetworks/nova-networks" # Path to the repository
DATADIR="/projectnb/btec-design3/novanetworks/kaggle_HPA/2021/data/bbbc036" # Path to the bbbc036 dataset
OPDATADIR="/projectnb/btec-design3/novanetworks/nova-networks/preprocessing/train_bbbc" # Path to the output directory
CELLPOSE_CONDAENV="cellpose2" # Conda environment name

SCRIPTDIR="${REPODIR}/preprocessing/BBBC036" # Path to preproc_bbbc.sh
IPDIR="${DATADIR}/raw/csv_unzip" # Path to the input images
NCPU=30 # Number of CPU cores
INPUTTYPE="train" # "train" or "valid"

#=================================================================================================================================

for type in $INPUTTYPE; do

    OPIMGDIR="${OPDATADIR}/img" # 4ch RGBA results
    OPIMGRGBDIR="${OPDATADIR}/imgRGB" # 3ch RGB results 
    OPSEGDIR="${OPDATADIR}/seg" # Cellpose seg results (in .png & .npy)
    OPCELLDIR="${OPDATADIR}/cell" # Cropped cells
    OPCODEDIR="${OPDATADIR}/mask" # Encoded masks for Kaggle Challenge Submission
    CSVPATH="${REPODIR}/HPA-nova/dataloaders/split/${type}_bbbc.csv"
    FRAMESIZE=1024 # examples: "1024", "1024 1024", or "1024 512"

    
    # Check if directories exist, if not, create them
    if [ ! -d "$OPIMGDIR" ]; then
        mkdir -p "$OPIMGDIR"
    fi

    if [ ! -d "$OPIMGRGBDIR" ]; then
        mkdir -p "$OPIMGRGBDIR"
    fi

    if [ ! -d "$OPSEGDIR" ]; then
        mkdir -p "$OPSEGDIR"
    fi

    if [ ! -d "$OPCELLDIR" ]; then
        mkdir -p "$OPCELLDIR"
    fi

    
    # Export 4ch RGBA 8-bit png 
    python "${SCRIPTDIR}/channel_merge_bbbc.py" \
       -i "${IPDIR}" \
       -ic "${CSVPATH}" \
       -o "${OPIMGDIR}" \
       -n $NCPU \
       -s $FRAMESIZE
    

    # Export 3ch RGB  8-bit png
    python "${SCRIPTDIR}/channel_merge_bbbc.py" \
        -i "${IPDIR}" \
        -ic "${CSVPATH}" \
        -o "${OPIMGRGBDIR}" \
        -n $NCPU \
        -s $FRAMESIZE \
        -m # Merge the ERSytoBleed and ERSyto channel


    # Get image statistics
    python "${SCRIPTDIR}/getstat.py" \
        -i "${OPDATADIR}/img" \
        -ic "${CSVPATH}" \
        -oc "${OPDATADIR}/stats" \
        -n $NCPU 


    # Create segmentation masks using cellpose
    conda activate ${CELLPOSE_CONDAENV}
    python "${SCRIPTDIR}/cellpose_seg.py" \
        -i "${OPIMGRGBDIR}" \
        -o "${OPSEGDIR}"\
    
    conda deactivate


    # Crop 4ch images using 3ch seg masks
    python "${SCRIPTDIR}/crop_img_bbbc.py" \
        -ic "${CSVPATH}" \
        -ii "${OPIMGDIR}" \
        -is "${OPSEGDIR}" \
        -o "${OPCELLDIR}" \
        --opcodedir "${OPCODEDIR}"\
        -n $NCPU 
        #--max_cell_count all # default: 10 crops 
done