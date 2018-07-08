# Style Change Detection

## Install dependencies
  * ```pip3 install -r requirements.txt```

## Optional dependencies
  * ```pip3 install pydot``` (keras model visualization)
  * ```apt install graphviz``` (keras model visualization)
  * ```pip3 install jupyter```
  * ```pip3 install h5py``` (saving keras models to disk)
  * ```pip3 install textstat```

## External resources / Prerequisites
  * Add training/validation data to ```data/training/``` / ```data/validation/``` (from https://pan.webis.de/clef18/pan18-web/author-identification.html)
  * Add evaluation script for the Style Change Detection task as ```pan18_scd_evaluator.py``` (from https://pan.webis.de/clef18/pan18-web/author-identification.html)
  * Add google books common words dictionary as ```google-books-common-words.txt``` to ```data/external/common_words/``` (from http://norvig.com/google-books-common-words.txt)
  * Add evaluation scripts for the Style Breach Detection task as ```windowdiff.py``` and ```winpr.py``` to ```metrics/``` (from https://pan.webis.de/clef17/pan17-web/author-identification.html)
  * Add pre-trained vectors to ```data/vectors/```
  * Add file ```config.json``` to root with:
  ```
  {
    "persist_results": true,
    "results_file": "results/my_results_file.json",
    "n_jobs": 1
  }
  ```
