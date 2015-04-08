package FeatureSelection;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import java.io.BufferedWriter;
import java.io.FileWriter;

import weka.core.matrix.Matrix;

public class RankFeatures {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {

		// Get the features file and read it
		String featuresFile = "./data/dexter_train.data";
		// open the features file
		BufferedReader brFeatures = new BufferedReader(new FileReader(featuresFile));

		// Get the labels file
		String labelsFile = "./data/dexter_train.labels";
		// open the labels file
		BufferedReader brLabels = new BufferedReader(new FileReader(labelsFile));

		// Create matrices
		Matrix mPEgs = new Matrix(150, 20000, 0);		// matrix positive class; 150 egs with 20000 features each
		Matrix mNEgs = new Matrix(150, 20000, 0);		// matrix negative class; 150 egs with 20000 features each
		Matrix mPMean = new Matrix(1, 20000, 0);		// Positive class mean matrix
		Matrix mNMean = new Matrix(1, 20000, 0);		// Negative class mean matrix

		// Declare string tokenizers for " " & ":"
		StringTokenizer spaceTokenizer;
		StringTokenizer colanTokenizer;

		// Now, fill these matrices based on the line info
		String fLine;	// features file line
		String lLine;	// labels file line
		boolean fileNotEnded = true;
		int pEgCount = -1;	// +ve example count
		int nEgCount = -1;	// -ve example count
		while(fileNotEnded) {
			fLine = brFeatures.readLine();
			lLine = brLabels.readLine();
			// If file ending reached then flag it
			if(fLine == null && lLine == null) {
				fileNotEnded = false;
				break;
			}

			// tokenize the line
			spaceTokenizer = new StringTokenizer(fLine, " ");

			// Check if the example is a +ve class 
			if(Integer.parseInt(lLine.trim()) == 1) {
				pEgCount++;
				while(spaceTokenizer.hasMoreTokens()) {
					// Tokenizing using space
					String fToken = spaceTokenizer.nextToken();
					// tokenize each feature and value
					colanTokenizer = new StringTokenizer(fToken, ":");
					// feature & value
					int featureNo = Integer.parseInt(colanTokenizer.nextToken())-1;	// feature value range from 0 to 19999 in array
					int featureValue = Integer.parseInt(colanTokenizer.nextToken());
					// fill it in matrix
					mPEgs.set(pEgCount, featureNo, featureValue);
					// sum the values in the mean matrix
					mPMean.set(0, featureNo, mPMean.get(0, featureNo)+featureValue);
				}
			}
			else if(Integer.parseInt(lLine.trim()) == -1){
				nEgCount++;
				while(spaceTokenizer.hasMoreTokens()) {
					// Tokenizing using space
					String fToken = spaceTokenizer.nextToken();
					// tokenize each feature and value
					colanTokenizer = new StringTokenizer(fToken, ":");
					// feature & value
					int featureNo = Integer.parseInt(colanTokenizer.nextToken())-1;	// feature value range from 0 to 19999 in array
					int featureValue = Integer.parseInt(colanTokenizer.nextToken());
					// fill it in matrix
					mNEgs.set(nEgCount, featureNo, featureValue);
					// sum the values in the mean matrix
					mNMean.set(0, featureNo, mNMean.get(0, featureNo)+featureValue);
				}
			}
			else {
				System.out.println("Invalid Labels! @ LineCount: " + (pEgCount+nEgCount+2));
				System.exit(0);
			}
		}
		System.out.println("Done!");

		// Divide the sum of values with no of examples to get the mean
		mPMean.times(pEgCount+1);
		mNMean.times(nEgCount+1);

		// Create the standard deviation matrices
		Matrix mPStandardDeviation = new Matrix(1, mPMean.getColumnDimension(), 0);
		Matrix mNStandardDeviation = new Matrix(1, mNMean.getColumnDimension(), 0);

		// Compute SD
		for(int i=0; i<mPEgs.getColumnDimension(); i++) {	// Iterate for each feature
			int sumOfPDiffs = 0;
			int sumOfNDiffs = 0;
			for(int j=0; j<mPEgs.getRowDimension(); j++) {	// iterate for each example (on the same feature)
				sumOfPDiffs += Math.pow(mPEgs.get(j, i) - mPMean.get(0, i), 2);
				sumOfNDiffs += Math.pow(mNEgs.get(j, i) - mNMean.get(0, i), 2);
			}
			double sdP = Math.sqrt(sumOfPDiffs/mPEgs.getRowDimension());	// standard deviation for +ve egs feature
			double sdN = Math.sqrt(sumOfNDiffs/mNEgs.getRowDimension());	// standard deviation for -ve egs feature
			mPStandardDeviation.set(0, i, sdP);								// set sd
			mNStandardDeviation.set(0, i, sdN);								// set sd
		}

		// Rank them using StoN Ratio
		List<Integer> ranksS2N =  RankWithS2N(mPMean, mNMean, mPStandardDeviation, mNStandardDeviation);
		// Rank with T-Test
		List<Integer> ranksTTest = RankWithTTest(mPMean, mNMean, mPStandardDeviation, mNStandardDeviation, pEgCount, nEgCount);
		// Rank with Pearson Correlation Coefficient
		List<Integer> ranksPCC = RankWithPCC(mPEgs, mNEgs, mPMean, mNMean);

		// Write ranks to files
		writeListToFile(ranksS2N, "./results/s2n.ranks");
		writeListToFile(ranksTTest, "./results/ttest.ranks");
		writeListToFile(ranksPCC, "./results/pcc.ranks");
		
		// Close the buffer reader
		brFeatures.close();
		brLabels.close();
	}

	/**
	 * Rank the features using Signal to Noise Ratio
	 * @param mPMean
	 * @param mNMean
	 * @param mPSD
	 * @param mNSD
	 * @return
	 */
	public static List<Integer> RankWithS2N(Matrix mPMean, Matrix mNMean, Matrix mPSD, Matrix mNSD) {

		// Feature Number & corresponding values
		List<Double> valuesList = new ArrayList<Double>();
		List<Integer> featureIdList = new ArrayList<Integer>();
		List<Integer> rankedList = new ArrayList<Integer>();
		List<Double> rankedListValues = new ArrayList<Double>();

		// compute the values for each feature
		//		int zeroCount = 0;	// -- Debug
		for(int i=0; i < mPMean.getColumnDimension(); i++) {
			double denominator = mPSD.get(0, i) + mNSD.get(0, i);
			double numerator = Math.abs(mPMean.get(0, i) - mNMean.get(0, i));
			if(denominator != 0) {
				valuesList.add(numerator/denominator);
				featureIdList.add(i);
			}
			else {
				//				zeroCount++;	// -- Debug
				valuesList.add(0.0);
				featureIdList.add(i);
			}
		}

		// -- Debug; Check for zero count
		//		System.out.println(zeroCount);

		// Sort the list
		for(int i=0; i < mPMean.getColumnDimension(); i++) {
			// Get the max value position to find the max feature id
			int maxValuePosition = valuesList.indexOf(Collections.max(valuesList));
			// Fill the corresponding max feature Id for ranking
			rankedList.add(featureIdList.get(maxValuePosition));
			// Fill the max value as well correspoding to its id
			rankedListValues.add(valuesList.get(maxValuePosition));

			// Remove the max feature value & id to find the next max elements
			valuesList.remove(maxValuePosition);
			featureIdList.remove(maxValuePosition);
		}

		// -- Debug
//		System.out.println(rankedList);
//		System.out.println(rankedListValues);

		// return the rank list
		return rankedList;
	}

	/**
	 * Rank the feature id's using T-Test
	 * @param mPMean
	 * @param mNMean
	 * @param mPStandardDeviation
	 * @param mNStandardDeviation
	 * @param pEgCount
	 * @param nEgCount
	 * @return
	 */
	public static List<Integer> RankWithTTest(Matrix mPMean, Matrix mNMean, Matrix mPSD, Matrix mNSD,
			int pEgCount, int nEgCount) {

		// Feature Number & corresponding values
		List<Double> valuesList = new ArrayList<Double>();
		List<Integer> featureIdList = new ArrayList<Integer>();
		List<Integer> rankedList = new ArrayList<Integer>();
		List<Double> rankedListValues = new ArrayList<Double>();

		// Calculate the T values for each and every feature
		for(int i=0; i<mPMean.getColumnDimension(); i++) {
			// compute numerator & denominator
			double numerator = mPMean.get(0, i) - mNMean.get(0, i);
			double denominator = Math.sqrt(Math.pow(mPSD.get(0, i), 2)/pEgCount +
					Math.pow(mNSD.get(0, i), 2)/nEgCount);
			if(denominator != 0) {
				valuesList.add(numerator/denominator);
				featureIdList.add(i);
			}
			else {
				valuesList.add(0.0);
				featureIdList.add(i);
			}
		}

		// Sort the list
		for(int i=0 ;i<mPMean.getColumnDimension(); i++) {
			// Get the max value position to find the max feature id
			int maxValuePosition = valuesList.indexOf(Collections.max(valuesList));
			// Fill the corresponding max feature Id for ranking
			rankedList.add(featureIdList.get(maxValuePosition));
			// Fill the max value as well corresponding to its id
			rankedListValues.add(valuesList.get(maxValuePosition));

			// Remove the max feature value & id to find the next max elements
			valuesList.remove(maxValuePosition);
			featureIdList.remove(maxValuePosition);
		}

		// -- Debug
//		System.out.println(rankedList);
//		System.out.println(rankedListValues);

		// return ranked list
		return rankedList;
	}


	public static List<Integer> RankWithPCC(Matrix mPEgs, Matrix mNEgs, Matrix mPMean, Matrix mNMean) {	// PCC = Pearson correlation coefficient
		// Feature Number & corresponding values
		List<Double> valuesList = new ArrayList<Double>();
		List<Integer> featureIdList = new ArrayList<Integer>();
		List<Integer> rankedList = new ArrayList<Integer>();
		List<Double> rankedListValues = new ArrayList<Double>();

		// According to the given dataset we have 150 training examples each for either classes.
		// So, Y Mean will become 0. And as we have classified examples for each class, we know its labels.
		// either +1 or -1. So, did not take them as parameters to this method.
		for(int i=0; i<mPEgs.getColumnDimension(); i++) {
			// compute numerator & denominator
			double numerator = 0 ;
			double denominator = 0;
			// Add up for each example
			for(int j=0; j<mPEgs.getRowDimension(); j++) {
				double pDeviation = (mPEgs.get(j, i) - mPMean.get(0,  i));
				double nDeviation = (mNEgs.get(j, i) - mNMean.get(0,  i));
				numerator += pDeviation*1 + nDeviation*-1;
				denominator += Math.pow(pDeviation, 2) + Math.pow(nDeviation, 2);
			}
			// We know in denominator Math.sqrt(sum of squares of deviations of labels) to be multiplied
			denominator = Math.sqrt(denominator) * Math.sqrt(300);	// 150 + 150 egs. (+1)^2 & (-1)^2

			// Fill the values in the values list
			if(denominator != 0) {
				valuesList.add(numerator/denominator);
				featureIdList.add(i);
			}
			else {
				valuesList.add(0.0);
				featureIdList.add(i);
			}
		}

		// Sort the list
		for(int i=0 ;i<mPMean.getColumnDimension(); i++) {
			// Get the max value position to find the max feature id
			int maxValuePosition = valuesList.indexOf(Collections.max(valuesList));
			// Fill the corresponding max feature Id for ranking
			rankedList.add(featureIdList.get(maxValuePosition));
			// Fill the max value as well corresponding to its id
			rankedListValues.add(valuesList.get(maxValuePosition));

			// Remove the max feature value & id to find the next max elements
			valuesList.remove(maxValuePosition);
			featureIdList.remove(maxValuePosition);
		}

		// -- Debug
//		System.out.println(rankedList);
//		System.out.println(rankedListValues);

		// return ranked list
		return rankedList;
	}
	
	/**
	 * Write given list to file
	 * @param list
	 * @throws IOException 
	 */
	private static void writeListToFile(List<Integer> list, String fileName) throws IOException {
		// open file to write
		BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
		for(int i=0; i<list.size(); i++) {
			bw.write(list.get(i).toString());
			bw.write("\n");
		}
		
		// Close buffer
		bw.close();
	}
}
