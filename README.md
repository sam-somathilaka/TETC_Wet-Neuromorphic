# TETC_Analyzing-Wet-Neuromorphic-Computing-Using-Bacterial-Gene-Regulatory-Neural-Networks

This study integrates bio-computing with neuromorphic systems by transforming bacterial gene regulatory networks into Gene Regulatory Neural Networks (GRNNs) for biocomputing applications. We analyze the intrinsic properties of gene regulations, convert them into a gene-perceptron function, and introduce a sub-GRNN search algorithm to tailor network structures for specific problems. Using Escherichia coli as the model organism, we extract and validate a base-GRNN for accuracy. Our comprehensive feasibility analysis demonstrates the GRNN’s computational capabilities in classification and regression tasks. Additionally, we explore the potential of applying GRNNs to digit classification as a use case. Simulation experiments indicate promising results for utilizing GRNNs in bacterial cells, thus advancing wet-neuromorphic computing with natural cells. This GitHub repository provides the required codes for the analysis.

## Data Files

This project requires the data files in the following list. However due to the file size restriction of Git, the codes are available upon request.

**1. InterploatedTranscriptomic.csv :**
This file includes the interpolated transcriptomic records for * *E. coli* * from the GEO Dataset (accession number GSE65244).

Structure:
|Time|g<sub>0</sub>|g<sub>1</sub>|...|g<sub>2</sub>|
|----|-------------|-------------|-------------|---|
|t<sub>0</sub> | e<sub>(0,0)</sub>    |e<sub>(0,1)</sub>|....|e<sub>(0,N)</sub>|
|...|....|...|...|...|
|t<sub>T</sub> | e<sub>(T,0)</sub>    |e<sub>(T,1)</sub>|...|e<sub>(T,N)</sub>|


**2. EColiGRN.csv :** 
This file contains the gene-gene interaction information extracted from the **regulondb** database.

**3. WeightMatrix.csv**
4. NormaizedBias.csv
5. InputGenes.csv

   **InterploatedTranscriptomic.csv:** 

## Weight Extraction

## Regression Analysis



## Classification Analysis

### Application-Specific sub-GRNN extraction

### Digit Classification use-case
