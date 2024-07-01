# Kmer Manifold Approximation and Projection (KMAP)
kmap is a package for visualizing kmers in 2D space. 
-![image](./kmap_cartoon.gif)

## Installation
```
conda env create -f environment.yml
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
  | -- kmer_count[folder]: kmer counts of different kmer lengths
  | -- candidate_conseq.csv: consensus sequences of significant Hamming balls of diferent kmer length
  | -- final_conseq.txt: merged consensus seqences of different kmer lengths, final motifs (2 motif in this case)
  | -- final_conseq.info.txt: contain meta information of final consensensus sequences
  | -- hamming_balls[folder]: Count matrix (calculated from Hamming balls) and logos of final motifs
  | -- co_occurence_mat.tsv: co-occurence matrix of final motifs
  | -- co_occurence_mat.norm.tsv: normalized co-occurence matrix of final motifs
  | -- motif_pos_density[folder]: motif kmer postion distribution on input sequences, position count from the reverse direction if reverse complement match the motif
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

[comment]: <> (Release commands)
[comment]: <> (python -m build) 
[comment]: <> (python3 -m twine upload --repository pypi dist/*)

