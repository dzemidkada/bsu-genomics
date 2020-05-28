#!/bin/bash

BAM_FILES_DIR="bam_files"
RESULT_DIR="loci_reads"
STR_MARKERS="str_codis_and_indels.txt"

rm -rf "$RESULT_DIR"

for sample in $(ls $BAM_FILES_DIR | awk -F. '{print $1}' | uniq | sort)
do
    echo "Processing: $sample"
    
    if [ -d "$RESULT_DIR/$sample" ]
    then
        echo "$sample has already been processed."
    else
        mkdir -p "$RESULT_DIR/$sample"
        
        while IFS= read -r line
        do
            locus_coordinates=$(echo $line | awk '{print $1}')
            locus=$(echo $line | awk '{print $2}')
            echo "Locus: $locus"

            echo "Retrieve locus reads..."
            ./samtools view -bh "$BAM_FILES_DIR/$sample.sorted.bam" $locus_coordinates > "$RESULT_DIR/$sample/$locus.bam"
            echo "Bam to Fq"
            ./samtools bam2fq "$RESULT_DIR/$sample/$locus.bam" > "$RESULT_DIR/$sample/$locus.fq"
            
            
            filesize=$(stat -f%z "$RESULT_DIR/$sample/$locus.fq")

            if [ $filesize -eq 0 ]; then
                rm -rf $RESULT_DIR/$sample/$locus.fq
            fi
            rm -rf $RESULT_DIR/$sample/$locus.bam

        done < "$STR_MARKERS"
    fi
done