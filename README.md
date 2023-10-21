# Neuron Spike Identification with Machine Learning

ECE 2024 Senior Design Project

## Authors
Victoria Carlsten (carlsten@bu.edu)\
Hao Chen (ha0chen@bu.edu)\
Claire Cropper (ccropper@bu.edu)\
Shi Gu (bengushi@bu.edu)

## Project Description 
### Client Information:
Chen Yang, PhD (cheyang@bu.edu)\
Vikrant Sharma (vikrant@bu.edu)

### Description of the Technical Problem:
We propose developing two machine learning algorithms that accurately identify neuron spikes in electrophysiology data such as LFP recordings. Neuron spikes indicate neural activity and offer clues about resting state connectivity and stimulus response, offering insights into understanding brain function. One algorithm will work retrospectively for analysis after experiment. The second algorithm will work in real-time for real-time spike recordings (RTsr) for data collection validation. This project aims to leverage machine learning techniques to automate and enhance retrospective and real time spike identification in complex electrophysiological recordings. 

### Methodology:
1. Identify robust public datasets of labeled electrophysiology data containing spike and non-spike segments.
2. Create preprocessing techniques to clean and transform raw public data into appropriate features for machine learning.
3. Design, train and optimize retrospective machine learning model such that performance (F1, accuracy, precision, recall) is similar to or better than conventional spike labeling models.
4. Evaluate and optimize machine learning model for data collected at Chen Yang Lab.
5. Develop app front-end for visualizing detected spikes and changing parameters
6. Repeat 1-5 for RTsr (5 being optional for RTsr)

### Expected Deliverables at the end of the course:
​Python software that accepts raw electrophysiology data and outputs labeled spike-trains and neuron clusters with confidence of labeling.\
​App front-end for visualizing spikes and altering parameters (such as clustering conductivity thresholds, accepted sensitivity, etc.).\
​Python software that accepts RTsr and labels spike-trains + neuron clusters in real time.\
​App front-end for real-time electrophysiology model (optional)
