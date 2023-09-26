[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8349582.svg)](https://doi.org/10.5281/zenodo.8349582)

(**add DOI for paper available**)


# gw190521-timedomain-release

This repository hosts scripts to download and plot the data in *"GW190521: tracing imprints of spin-precession on the most massive black hole binary"* - Miller et. al 2023 (**add arxiv link once available**).

Our dataset itself -- containing posterior samples -- is available to download from [Zenodo](https://doi.org/10.5281/zenodo.8349582).

In this `README` we describe how to download the posterior samples, run scripts to calculate supplementary data from the posterior samples (such as waveform reconstructions), and generate all the figures presented in the paper.

## Downloading data

After cloning this repository, to download our posterior samples, enter the `data` folder and enter the following: 
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

The `scripts` folder holds all the scripts and notebooks necessary to generate all supplementary data used to generate the paper's figures. 

First, to generate waveform reconstructions corresponding to our posterior samples: 
```
python reconstruct.py
```
This script generates the output file `waveform_reconstructions_all_detections.npy` in the `data`. folder. 
Note that this script takes hours to run; generaring waveform reconstructions is computationally costly. 
If the script crashes partway through for whatever reason, use the optional flag `--reload` to pick up where the code left off: 
```
python reconstruct.py --reload
```

Next, to track the time evolution of the inclination angle of the binary over time, run: 
```
python nrsur_angles.py
```
This script generates the output file `angles_vs_time_dict.npy` in the `data` folder.. It also has the optional `--reload` flag to pick up where the script left off in case of an incomplete calculation. 

Finally, run the script
```
calculate_SNRs.py
```
 to generate `snrs.npy` in the `data` folder, a dictionary storing the signal to noise ratios for the waveform in each detector corresponding to our posterior samples. 

## Making figures and gifs

In the `figures` folder, run each jupyter notebook to generate the corresponding figure (plus some supplemental figures available within each notebook). 

We have gifs corresponding to some figures to show our results at more timestamps than are shown in the paper. In the `gifs` folder, run each jupyter notebook to generate these gifs.

