#!/bin/bash

REF_INDEX="./../grch38_1kgmaj_bt2/grch38_1kgmaj"
FASTQ_FILES_DIR="fastq_data"
BAM_FILES_DIR="bam_files"

mkdir tmp_dir

for sample in $(ls $FASTQ_FILES_DIR | awk -F_ '{print $1}' | uniq | sort)
do
    echo "Processing: $sample"
    
    if [ -f "$BAM_FILES_DIR/$sample.sorted.bam" ]
    then
        echo "$sample has already been processed. Sorted BAM file found"
    else
        echo "\tUnzip fastq files..."
        SECONDS=0
        gunzip -c "$FASTQ_FILES_DIR/$sample""_L001_R1_001.fastq.gz" > "tmp_dir/$sample.R1.fastq"
        gunzip -c "$FASTQ_FILES_DIR/$sample""_L001_R2_001.fastq.gz" > "tmp_dir/$sample.R2.fastq"
        echo "$(($SECONDS / 60)) minutes and $(($SECONDS % 60)) seconds elapsed."

        echo "Generating sam file..."
        SECONDS=0
        ./bowtie2 -x $REF_INDEX -1 "tmp_dir/$sample.R1.fastq" -2 "tmp_dir/$sample.R2.fastq" -S "tmp_dir/tmp.sam"
        echo "$(($SECONDS / 60)) minutes and $(($SECONDS % 60)) seconds elapsed."

        echo "Transform sam into bam..."
        SECONDS=0
        ./samtools view -S -b "tmp_dir/tmp.sam" > "$BAM_FILES_DIR/$sample.bam"
        echo "$(($SECONDS / 60)) minutes and $(($SECONDS % 60)) seconds elapsed."

        echo "Sort bam file"
        SECONDS=0
        ./samtools sort "$BAM_FILES_DIR/$sample.bam" > "$BAM_FILES_DIR/$sample.sorted.bam"
        echo "$(($SECONDS / 60)) minutes and $(($SECONDS % 60)) seconds elapsed."

        echo "Index sorted bam file"
        SECONDS=0        
        ./samtools index "$BAM_FILES_DIR/$sample.sorted.bam"
        echo "$(($SECONDS / 60)) minutes and $(($SECONDS % 60)) seconds elapsed."

        echo "Remove temp data"
        rm -rf "tmp_dir/$sample.R1.fastq" "tmp_dir/$sample.R2.fastq" "tmp_dir/tmp.sam" "$BAM_FILES_DIR/$sample.bam"
    fi
    
done