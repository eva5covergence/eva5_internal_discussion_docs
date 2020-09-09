**Only files need to update for any specific project are:**

- configs/basic_config.py - Config file
- models/Networks/*.py - mnist_specific config - architecture
- Orchestrator/*.py 
- Final Ipynb file in the root folder which triggers main.py
- Show the architecture of the network in main ipynb file


**Issues on google collab that to be fixed:**

1. Unable to see Tqdm progress and other print statements on ipynb, unable capture tqdm progress in logs
2. Unable to see plotted graphs
3. Add more logging statements wherever it is needed
4. Check any other issues while executing
5. Session7 specific assignment on advanced convolutions
6. Input the dataset name in orchestrator rather than in basic_config
7. ipynb file should download git repo to google drive rather not temporary space provided by google collab

**Later pending parts:**

- Add a method which returns receptive field for each layer of network
- Unit tests
- Add docstrings and Sphinix Documentation 
- CICD pipelines
- Make it distributable and installable - Fix setup.py















