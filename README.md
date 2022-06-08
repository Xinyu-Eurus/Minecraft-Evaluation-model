# Minecraft-Evaluation-model

## To developer 

Please put the "models" folder under the same directory with 'classifier.py'. 
Then run the 'classifier.py'.
The demo_predict mode can only predict labels one by one. 

demo_data.npy is one episode of demo data collected from the player.
The size and features are as the descriptions.pdf shows.
The "models" folder includes 7 models:
	* km_0.model to km_4.model is for kmeans clustering of feature[0] to feature[4];
	* RF-origin.model is the random forest model trained by original data-label pairs;
	* RF-optim.model is the random forest model trained by optimized data-label pairs, the average accuracy in testing is usually 0.01-0.03 higher than RF-origin.model.



## To Shihan

* "data_origin.npy" includes all data and labels that simply count major voted labels. If there are more than one labels got the highest votes, choose the smallest label.
e.g. if there is an episode got [2,2,1] votes, the label is "0".
Shape: N rows (N=number of data episode), 21 columns (column 0-19 are features, column 20 is extracted label)

* "data_optim.npy" includes all data and labels that deleted multi-major labels and relevant data.
Shape: The same

*Evaluating by the confidence didn't get as good results as data_origin and data_optim, so I didn't save it. But can still find in "model-exploring.ipynb".
