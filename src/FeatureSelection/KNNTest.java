package FeatureSelection;

import java.io.BufferedReader;
import java.io.FileReader;
//import java.io.IOException;

import weka.core.Instances;
//import weka.core.Instance;
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;

public class KNNTest {

	/**	 Params	 **/
	// File for Ranking
	public static String rankingAlgorithm = "pcc";	// valid values: s2n; ttest; pcc
	// If the feature vector to be normalized or not
	public static boolean normalize = true;			// false for not normalizing
	// type of data	- train or validation
	public static String typeOfData = "train"; 		// train ; valid
	// k value - in k-NN
	public static int k = 10; 						// 1 - 5 - 10

	// Main Method
	public static void main(String[] args) throws Exception {

		// No of top Features
		int topN[] = {1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000,
				8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000};

		for(int i=0; i<topN.length; i++) {
			// current topN value
			int noOfTopFeatures = topN[i];
			
			/** Load the instances into classifier **/
			// Open Training File
			String fileName = "./data/"+ rankingAlgorithm + "_rank" + noOfTopFeatures + "_Norm" + normalize + "_weka.train";
			BufferedReader brTrain = new BufferedReader(new FileReader(fileName));
			// Load the training instances
			Instances trainingData = new Instances(brTrain);
			trainingData.setClassIndex(trainingData.numAttributes()-1);
			// close the buffer
			brTrain.close();
			
			/** Load the test instances into classifier **/
			// Open test file
			fileName = "./data/"+ rankingAlgorithm + "_rank" + noOfTopFeatures + "_Norm" + normalize + "_weka.valid";
			BufferedReader brTest = new BufferedReader(new FileReader(fileName));
			// Load the training instances
			Instances testData = new Instances(brTest);
			testData.setClassIndex(trainingData.numAttributes()-1);
			// close the buffer
			brTest.close();
			
			/** Create a classifier and add the training data **/
			Classifier ibk = new IBk(k);
			ibk.buildClassifier(trainingData);
			
			/** Test & Print Accuracy **/
			int totalTestInstancesCount = testData.numInstances();
			int totalCorrectPredictions = 0;
			for(int n=0; n<totalTestInstancesCount; n++) {
				int prediction = (int) Double.parseDouble(testData.classAttribute().value((int) ibk.classifyInstance(testData.instance(n))));
				int original = (int) Double.parseDouble(testData.classAttribute().value((int) testData.instance(n).classValue()));
				if(prediction == original)
					++totalCorrectPredictions;
			}
			System.out.println(topN[i]+"- Accuracy: "+(100*totalCorrectPredictions/(double)totalTestInstancesCount)+ " " + totalCorrectPredictions + "/" + totalTestInstancesCount);
		}
	}
}
