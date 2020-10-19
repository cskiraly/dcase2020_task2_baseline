baseline_toco_tf23rc2
: DCASE2020 baseline model trained and converted with the older TOCO converter of TFLite 2.3-rc2. Best version.

baseline_tf23rc2
: DCASE2020 baseline model trained and converted with default TFLite 2.3-rc2 converter. This version is missing the folding of BatchNorm layers, don't use!

baseline_folded_tf23rc2
: DCASE2020 baseline model trained. Manual BatchNorm folding applied before TFLite 2.3-rc2 conversion. Almost as good as the TOCO version.

