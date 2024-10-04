# Kmer Manifold Approximation and Projection (KMAP)
kmap is a package for visualizing kmers in 2D space. 
-![image](./kmap_cartoon.gif)

## Installation

```
conda create --name=kmap_test python=3.11
conda activate kmap_test
conda install anaconda::scipy
conda install anaconda::numpy
conda install anaconda::matplotlib
conda install anaconda::pandas
conda install anaconda::click
conda install anaconda::tomli-w
conda install anaconda::requests
conda install conda-forge::biopython
conda install conda-forge::networkxx
conda install bioconda::logomaker
pip install taichi
pip install kmer-map
```

OR

```Mac
# Mac
conda env create -f environment.yml
conda activate kmap_test
```

OR

```Linux
# Linux
conda env create -f env.yml   
conda activate kmap_test
pip install -r requirements.txt
```


## Example usage
### General workflow
The following code shows the typical workflow of the test data in `./tests/test.fa`
```bash
# step 0: preprocess input fast file
kmap preproc --fasta_file ./tests/test.fa --res_dir ./test 

# step 1: scanning for motifs 
kmap scan_motif --res_dir ./test --debug true

# step 2: visualize kmers
# now edit the "./test/config.toml" file, in the 3rd section "visualization"
# change "n_max_iter = 2500" to "n_max_iter = 100"  
kmap visualize_kmers --res_dir ./test --debug True
```



### Detailed workflow
- **Step 0**: confirm kmap is successfully installed
```bash
kmap --help
```
- **Step 1**: create a directory `test` and save the test fasta file [test.fa](./tests/test.fa) in this directory
```bash 
mkdir ./test
```
The output folder looks like
```
test
  | -- test.fa
```
- **Step 2**: preprocess input fast file
```bash 
kmap preproc --fasta_file ./test/test.fa --res_dir ./test
```
The output folder looks like
```
test
  | -- test.fa
  | -- input.bin.pkl
  | -- input.seqboarder.bin.pkl
  | -- config.toml
  | -- motif_def_table.csv
```
`input.bin.pkl` is the processed `.pkl` file of the input fasta file `test.fa`, which can be read by the pickle module of python. 
`input.seqboarder.bin.pkl` is `n_seq x 2` numpy matrix, where the rows index each sequence, and the columns are start and end of each sequence. 
`config.toml` contains all the kmap parameters for this analysis task. You can modify the parameters according to your own needs.
`motif_def_table.csv` is the motif definition table, which specify the parameters of the Hamming balls. You can modify the parameters according to your own needs.
- **Step 3**: scan for motifs
```bash 
kmap scan_motif --res_dir ./test --debug true 
```
The `--debug` option can be omitted. The output folder with file/folder descriptions:
```
test
  | -- test.fa
  | -- input.bin.pkl
  | -- input.seqboarder.bin.pkl
  | -- config.toml
  | -- motif_def_table.csv
  | -- kmer_count[folder]: kmer counts of different kmer lengths, motif occurence file for candidate consensus sequences
  | -- candidate_conseq.csv: consensus sequences of significant Hamming balls of diferent kmer length
  | -- final_conseq.txt: merged consensus seqences of different kmer lengths, final motifs (2 motif in this case)
  | -- final_conseq.info.csv: contain meta information of final consensensus sequences
  | -- final.motif_occurence.csv: positions of final conseqs in the input reads 
  | -- conseq_similarity[folder]: files illustrate similarities between final conseqs
  | -- hamming_balls[folder]: Count matrix (calculated from Hamming balls) and logos of final motifs
  | -- co_occurence[folder]: files about co-occurence of final conseqs, e.g., co-occurence frequency, distance distribution between different conseqs
  | -- motif_pos_density[folder]: motif kmer postion distribution on input sequences
  | -- motif_pos_density.np.pkl: numpy array of the position densities
  | -- sample_kmers.pkl: sampled kmers for visualization
  | -- sample_kmers.tsv: sampled kmers for visualization, second column is final motif label, largest label (2) means random kmers
  | -- sample_kmer_hamdist_mat.pkl: Hamming distance matrix of sampled kmers, used as input for kmer visualization
```
-  **Step 4**: visualize kmers
Now now edit the `./test/config.toml` file, 
in the 3rd section `visualization` change `n_max_iter = 2500` to `n_max_iter = 100`. 
This will reduce the optimization steps from 2500 to 100, which saves lots of running time.
```bash 
kmap visualize_kmers --res_dir ./test --debug True
```
This will generate three additional files in the output directory
```
test
  | -- low_dim_data.tsv: low dimensional embeddings, the columns are (x, y, motif_label), where the largest label is random kmers.
  | -- ld_data.pdf: 2d plot of the embeddings in pdf 
  | -- ld_data.png: 2d plot of the embeddings in png
```

### Auxiliary functions

Sometimes we may want to adjust the default workflow.
For example, after checking the motif logos in the `hamming_balls` directory, we feel the consensus is `AATCGATAGC`, instead of `A[AATCGATAGC]GA`. 
We can re-generate the motif use the following commands.
```bash
kmap ex_hamball --res_dir ./test --conseq AATCGATAGC --return_type matrix --output_file ./test/hamming_balls/AATCGATAGC_cntmat.csv 
kmap draw_logo --cnt_mat_numpy_file ./test/hamming_balls/AATCGATAGC_cntmat.csv --output_fig_file ./test/hamming_balls/logo.pdf
```
The motif logo for this consensus sequence is given by `./test/hamming_balls/logo.pdf`

For example, we feel the maximum number of mutations `max_ham_dist=5` for the consensus sequence `GTACGTAGGTCCTA` (defined in `motif_def_table.csv`, k=14) is too large.  
We want to change the number of maximum mutations to 3. We can derive the new motif based on `max_ham_dist=5` using the following commands.
```bash
kmap ex_hamball --res_dir ./test --conseq GTACGTAGGTCCTA --return_type matrix --max_ham_dist 3 --output_file ./test/hamming_balls/GTACGTAGGTCCTA_cntmat.csv 
kmap draw_logo --cnt_mat_numpy_file ./test/hamming_balls/GTACGTAGGTCCTA_cntmat.csv --output_fig_file ./test/hamming_balls/logo.pdf
```
The motif logo for this consensus sequence is given by `./test/hamming_balls/logo.pdf`

It take efforts to compare the consensus sequences, especially when reverse complements are considered.
We can check the similarities between candidate consensus sequences using the following command.
```bash
kmap align_conseq --conseq_csv_file ./test/candidate_conseq.csv --out_dir ./test/candidate_seq_similarity
# or only final conseqs
kmap align_conseq --conseq_csv_file ./test/ final_conseq.info.csv --out_dir ./test/conseq_similarity
```
A dendrogram file `dendrogram.pdf` will be generated in the output directory `./test/candidate_seq_similarity`. 
The dendrogram shows the similarities between all candidate consensus sequences, as well as their reverse complements.
Hierarchical clustering is performed on the dendrogram. For each derived cluster, local pairwise alignment is generated
for each pair of consensus sequences in that cluster.

In Chip-seq data, we generally have the peak bed files, then we extract the corresponding fasta file from these peaks
and perform kmap analysis. After that, we would like to know the actual locations of detected motifs on the reference genome.
We could use the following command:
```bash
cd ./test # change to the result directory
kmap extract_motif_locations --bed_file your_bed_file.bed 
```
A new folder `motif_locations` will be generated in the result directory, which contains the actual genomic location of
detected final motifs.

[comment]: <> (Release commands)
[comment]: <> (python -m build) 
[comment]: <> (python3 -m twine upload --repository kmer-map dist/*)

