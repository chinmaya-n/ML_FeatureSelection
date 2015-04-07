package FeatureSelection;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;

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
		String labelsFile = "./data/dexeter_train.labels";
		// open the labels file
		BufferedReader brLabels = new BufferedReader(new FileReader(labelsFile));

		// Create matrices
		Matrix mPClass = new Matrix(150, 20000, 0);	// matrix positive class
		Matrix mNClass = new Matrix(150, 20000, 0);	// matrix negative class
		Matrix mPMean = new Matrix(1, 20000, 0);		// Positive class mean matrix
		Matrix mNMean = new Matrix(1, 20000, 0);		// Negative class mean matrix

		// Declare string tokenizers for " " & ":"
		StringTokenizer spaceTokenizer;
		StringTokenizer colanTokenizer;

		// Now, fill these matrices based on the line info
		String fLine;	// features file line
		String lLine;	// labels file line
		boolean fileNotEnded = true;
		int lineCount = 0;
		while(fileNotEnded) {
			fLine = brFeatures.readLine();
			lLine = brLabels.readLine();
			++lineCount;
			// If file ending reached then flag it
			if(fLine == null && lLine == null) {
				fileNotEnded = false;
			}

			// tokenize the line
			spaceTokenizer = new StringTokenizer(fLine, " ");

			// Check if the example is a +ve class 
			if(Integer.parseInt(lLine.trim()) == 1) {
				while(spaceTokenizer.hasMoreTokens()) {
					// Tokenizing using space
					String fToken = spaceTokenizer.nextToken();
					// tokenize each feature and value
					colanTokenizer = new StringTokenizer(fToken, ":");
					// fill it in matrix
					mPClass.set(lineCount, Integer.parseInt(colanTokenizer.nextToken()),
							Integer.parseInt(colanTokenizer.nextToken()));
				}
			}
			else {
				
			}
		}
	}

}
