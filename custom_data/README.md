# Source Codes

## Processing of Human Protein Atlas data
This is done in `hpa_analysis.ipynb` in the `src` directory. 

1. The 'subcellular location data' from [Human Protein Atlas](https://www.proteinatlas.org/about/download) is first downloaded and renamed as `subcellular_location.csv`. 
2. This data is then processed where the rows with `Uncertain` reliability are removed.
3. The `Uncertain` locations are removed as well.
4. Only `Gene`, `Reliability` and `Locations` are then stored in the `hpa.csv` file in the `data` directory. The `Gene` column stores the Gene Ensembl ID.

## Getting the Right Format of STRING data
This is done using the `txt2csv.py` in the `src` directory. 

1. The [STRING](https://string-db.org/cgi/download.pl) data of 'human' organism is first downloaded. This file is named `9606.protein.links.detailed.v11.0.txt` 
2. It is run through the `txt2csv.py` to get `string_human.csv` file in the `data` directory. __This file is not in the repository because of the size restrictions.__

## Processing the STRING data
This is done in `process_string.ipynb` in the `src` directory.

1. The `string_human.csv` is loaded from the `data` directory. 
2. We remove the rows that have duplicate sets of proteins -- {protein1, protein2}.
3. This new data is now stored as `string_clean.csv` in the `data` directory. __This file is not in the repository because of the size restrictions.__

## Mapping the Protein Ensembl IDs to Gene Ensembl IDs
This is done using the `mappings.ipynb` in the `src` directory. We first tried getting the mapping between the two IDs using [mygene](https://docs.mygene.info/en/latest/doc/packages.html#mygene-python-module) library in Python. But this did not give us any mappings so we shifted to the web version of [Biomart](https://m.ensembl.org/biomart/martview/).

1. Run the `mappings.ipynb` cell by cell.
2. We load `string_clean.csv` and get all the unique protein Ensembl IDs. These are saved in `prots.txt` in the `data` directory.
3. We then move to the [Biomart](https://m.ensembl.org/biomart/martview/) server.
    1. Select __Ensembl Genes 100__ Database.
    2. Select __Human genes (GRCh38.p13)__ Dataset.
    3. Now click on __Filters__ in the right side menu.
    4. Go to __GENE ->  Input external references ID list (Max 500 advised)__.
    5. Select __Protein stable IDs__ and upload the `prots.txt` from `data` directory.
    6. Now, click on __Attributes__ in the right side menu.
    7. Go to __GENE__ and select __Protein stable ID__ and __Gene stable ID__. 
    8. Click on __Results__ from the top menu.
    9. You will now see a screen with 10 mappings. In the top you will see __TSV__ at one place. Replace that with __CSV__ and click on the __GO__ button.
    10. Save this file as `mart.csv` in the `data` directory.
4. We now create a dictionary of this mapping where 'key' is the protein Ensembl ID and 'value' is the gene Ensembl ID.
5. This is then pickled in a file called `prot2gene.pickle` in the `data` directory.

## Combining HPA and STRING data
This is done using the `combine.ipynb` file in the `src` directory.

1. Load `hpa.csv`, `string_clean.csv` and `prot2gene.pickle` from the `data` directory.
2. Create a mapping from gene Ensembl ID to a tuple of (locations, reliability) using the HPA data.
3. Iterate through the STRING data and annotate each row that has at least one protein Ensembl ID that maps to gene Ensembl ID that exists in the HPA data.
4. Drop the remaining rows.
5. Format the locations in the list format.
6. The final data file is stored as `string_locs2.csv` in the `data` directory. __This file is not in the repository because of the size restrictions.__ 

The final data can be found [here](https://drive.google.com/file/d/1o3gvzdcqLgZ5O0alFoqtEhXL0YvXjuDr/view?usp=sharing) instead.

## Processing BioPlex data
The BioPlex data can be downloaded from [here](https://bioplex.hms.harvard.edu/data/BioPlex_293T_Network_10K_Dec_2019.tsv). The processing is done in the `bioplex.ipynb` file in a similar way as done for the STRING database. 

## Getting Ensembl IDs for BioPlex data
This is done using files `bioplex_unique_genes.ipynb` and `clean_biomart2ensemblt.ipynb`. The first file extracts the unique genes from the above processed file. This is saved as a text file in the `data` directory. This will be used to generate ensembl IDs using Uniprot. 

This is then processed through [Uniprot](https://www.uniprot.org/uploadlists/) to get ensembl transcript IDs. This is then passed through the Biomart as explained before to extract ensembl gene IDs and protein IDs. This is then mapped with genes in the BioPlex data using `clean_biomart2ensemblt.ipynb`.

This processed BioPlex data file can be found in the `data` directory. 

## Combining processed BioPlex and STRING data
We then combine the processed BioPlex and STRING data using `combine_bioplex_string.ipynb` in the `src` directory. We retain the Bioplex locations in case of any disagreements in the two datasets. 

We then remove the following locations - 

* Rods & Rings
* Aggresome
* Microtubule ends
* Cleavage furrow

These locations are removed either because there are very few datapoints for them and they are independent according to the *GO cellular_component* ontology or if they were not found in the ontology altogether. 

The final dataset can be found [here](https://drive.google.com/file/d/1uVRoAZFNjormaa496YLwd3nYpoiwArbv/view?usp=sharing). It consists of nearly 3.2 million datapoints and 28 different subcellular sites.
