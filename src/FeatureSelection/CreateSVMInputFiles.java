package FeatureSelection;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import weka.core.matrix.Matrix;

public class CreateSVMInputFiles {

	/**	 Params	 **/
	// File for Ranking
	public static String rankingAlgorithm = "pcc";	// valid values: s2n; ttest; pcc
	// If the feature vector to be normalized or not
	public static boolean normalize = false;			// false for not normalizing
	// type of data	- train or validation
	public static String typeOfData = "valid"; 		// train ; valid

	public static void main(String[] args) throws IOException {

		/** Get the training examples along with its features & Labels **/
		// Get the features file and read it
		String featuresFile = "./data/dexter_"+ typeOfData +".data";
		// open the features file
		BufferedReader brFeatures = new BufferedReader(new FileReader(featuresFile));

		// Get the labels file
		String labelsFile = "./data/dexter_"+ typeOfData +".labels";
		// open the labels file
		BufferedReader brLabels = new BufferedReader(new FileReader(labelsFile));

		// Create matrices
		Matrix trainingEgs = new Matrix(20000, 300, 0);		// 300 egs with 20000 features each
		Matrix trainingLabels = new Matrix(1, 300, 0);

		// Declare string tokenizers for " " & ":"
		StringTokenizer spaceTokenizer;
		StringTokenizer colanTokenizer;

		// Now, fill these matrices based on the line info
		String fLine;	// features file line
		String lLine;	// labels file line
		boolean fileNotEnded = true;

		int egCount = 0; 	// example count variable
		// Iterate over each line
		while(fileNotEnded) {
			// Read features line
			fLine = brFeatures.readLine();
			lLine = brLabels.readLine();
			// If file ending reached then flag it
			if(fLine == null && lLine == null) {
				fileNotEnded = false;
				break;
			}

			// tokenize the line
			spaceTokenizer = new StringTokenizer(fLine, " ");

			// Iterate over each feature and store the values
			while(spaceTokenizer.hasMoreTokens()) {
				// Tokenizing using space
				String fToken = spaceTokenizer.nextToken();
				// tokenize each feature and value
				colanTokenizer = new StringTokenizer(fToken, ":");
				// feature & value
				int featureNo = Integer.parseInt(colanTokenizer.nextToken())-1;	// feature value range from 0 to 19999 in array
				double featureValue = Double.parseDouble(colanTokenizer.nextToken());
				// fill it in matrix
				trainingEgs.set(featureNo, egCount, featureValue);
			}

			// Iterate each line and save into labels matrix
			trainingLabels.set(0, egCount, Double.parseDouble(lLine));

			// increment the example count
			egCount++;
		}
		System.out.println("Done reading examples: " + egCount);
		// test
		// saveMatrixToFile(trainingEgs, "./data/inputdata.matrix");
		// close the buffers
		brFeatures.close();
		brLabels.close();

		// Get the feature vector id's in ranked order based on 
		// Filter Method used.
		Matrix featuresRanked = new Matrix(20000, 1, 0);
		BufferedReader brRanked = new BufferedReader(new FileReader("./results/"+ rankingAlgorithm +".ranks"));
		String rankLine = brRanked.readLine();
		for(int i=0; i<featuresRanked.getRowDimension(); i++) {
			featuresRanked.set(i, 0, Double.parseDouble(rankLine));
			rankLine = brRanked.readLine();
		}
		// close the buffers
		brRanked.close();

		// List of "No of Top 'N' ranked features" to be used to generate files
		int topN[] = {1, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000,
				8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000};

		/** Change code below this portion to convert to some other file format like arff etc**/
		// Create Batches of threads
		int noOfBatches = 7;
		for(int b=0; b<noOfBatches; b++) {
			int threadsPerBatch = 5;
			if(b==6)
				threadsPerBatch = 4;
			
			// initialize the thread service
			final ExecutorService service = Executors.newFixedThreadPool(threadsPerBatch);
			// List to store the return values for each thread
			List<Future<int[]>> threadList = new ArrayList<Future<int[]>>();
			
			// Start the threads
			for(int n=0; n<threadsPerBatch; n++) {
				threadList.add(service.submit(new CreateThread(topN[b*5+n], trainingEgs, trainingLabels, featuresRanked,
						rankingAlgorithm, normalize, typeOfData)));
			}
			
//			// Dummy Wait for the threads to run completely
//			// total aggregate accuracy counts
//			int[] returnValues = {0, 0};
//			// Aggregate the accuracies
//			try{
//				for(int n=0; n<threadsPerBatch; n++) {
//					returnValues[0] += threadList.get(n).get()[0];
//					returnValues[1] += threadList.get(n).get()[1];
//				}
//			} catch(final InterruptedException ex) {
//				ex.printStackTrace();
//			} catch(final ExecutionException ex) {
//				ex.printStackTrace();
//			}
//			// Dummy Wait --- Not needed! Just to used for testing
			
			// shutdown service
			service.shutdown();
		}
	}

	// Print matrix
	public static void saveMatrixToFile(Matrix m, String fileName) throws IOException {
		// open file
		BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));
		for(int row=0; row<m.getRowDimension(); row++) {
			for(int col=0; col<m.getColumnDimension(); col++) {
				bw.write(m.get(row, col)+" ");
			}
			bw.write("\n");
		}

		// close buffer
		bw.close();
	}
}

// Thread Creation
class CreateThread implements Callable<int[]> {
	
	// member variables
	int N;
	Matrix trainingEgs;
	Matrix trainingLabels;
	Matrix featuresRanked;
	String rankingAlgorithm;
	boolean normalize; 
	String typeOfData;
	
	// constructor
	public CreateThread(int N, Matrix trainingEgs, Matrix trainingLabels, Matrix featuresRanked,
			String rankingAlgorithm, boolean normalize, String typeOfData) {
		this.N = N;
		this.trainingEgs = trainingEgs;
		this.trainingLabels = trainingLabels;
		this.featuresRanked = featuresRanked;
		this.rankingAlgorithm = rankingAlgorithm;
		this.normalize = normalize;
		this.typeOfData = typeOfData;
	}
	
	// init function
	public int[] call() throws IOException {
		/** Change code below this portion to convert to some other file format like arff etc**/
		// Iterate over no of ranked features to keep in the file
		// & Generate files
		int noOfTopFeatures = N;	// top n features

		// Create file to write
		String fileName = "./data/"+ rankingAlgorithm + "_rank" + noOfTopFeatures + "_Norm" + normalize + "_svm." + typeOfData;
		BufferedWriter bw = new BufferedWriter(new FileWriter(fileName));

		// iterate over no. of examples (i.e each line in the output file)
		for(int eg=0; eg<trainingEgs.getColumnDimension(); eg++) {
			String line = "";	// line to add to the file
			// Add the label for this example
			line = line + trainingLabels.get(0, eg);

			// Create a matrix to store the feature vector & for normalization
			Matrix featureVector = new Matrix(noOfTopFeatures, 1, 0);
			// Add the features to the feature vector of a particular example
			for(int f=0; f<noOfTopFeatures; f++) {
				double featureId = featuresRanked.get(f, 0);
				double featureValue = trainingEgs.get((int)featureId, eg);
				featureVector.set(f, 0, featureValue);
			}
			// Normalize the featureVector if needed
			if(normalize) {
				double norm = featureVector.normF();
				//					saveMatrixToFile(featureVector, "./results/featureVector.matrix");
				if(norm!=0)
					featureVector = featureVector.times(1/norm);
				else
					featureVector = new Matrix(featureVector.getRowDimension(), featureVector.getColumnDimension(), 0);
			}

			// Add the features to the line
			for(int i=0; i<featureVector.getRowDimension(); i++) {
				double fValue = featureVector.get(i, 0);
				if(fValue != 0) {
					line = line + " " + (i+1) + ":" + fValue;
				}
			}

			// end the line
			line = line + "\n";
			// write this line to the file
			bw.write(line);
		}

		// close buffers
		bw.close();
		// Wrote to file
		System.out.println("Wrote file: "+fileName);
		
		// return - dummy return
		int ret[] = {0, 1};
		return ret;
	}
}
