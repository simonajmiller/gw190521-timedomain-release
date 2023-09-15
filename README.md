[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8349582.svg)](https://doi.org/10.5281/zenodo.8349582)
[add DOI for paper once available]


# gw190521-timedomain-release

This repository hosts scripts to download and plot the data in "GW190521: tracing imprints of spin-precession on the most massive black hole binary" (Miller et. al 2023) [WORK IN PROGRESS: add arxiv link once available]

[insert brief description of paper and what the dataset includes]

Our dataset itself is available to download from [Zenodo](https://doi.org/10.5281/zenodo.8349582).
In this `README` we describe how to download the posterior samples, run scripts to calculate supplementary data from the posterior samples (such as waveform reconstructions), and generate all the figures presented in the paper.

## Downloading data

To download our posterior samples, enter the `data` folder and enter the following: 
```
chmod +x download_data_from_zenodo.sh
./download_data_from_zenodo.sh
```

Then, to download relevant LVC data (strain and posterior samples): 
```
cd GW190521_data
chmod +x get_data.sh
./get_data.sh
```

## Generating waveform reconstructions, etc. 


## Making figures
