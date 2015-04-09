import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
 
import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;
 
public class knn_test {
	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
 
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
 
		return inputReader;
	}
 
	public static void main(String[] args) throws Exception {
		BufferedReader datafile = readDataFile("./data/ads.txt");
 
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
 
		//do not use first and second
		Instance first = data.instance(0);
		Instance second = data.instance(4);
//		data.delete(0);
//		data.delete(4);
 
		Classifier ibk = new IBk();		
		ibk.buildClassifier(data);
 
		double class1 = ibk.classifyInstance(first);
		double class4 = ibk.classifyInstance(second);
 
		System.out.println("first: " + class1 + "\nsecond: " + class4);
		System.out.println("first true: " + data.classAttribute().value((int) data.instance(0).classValue()) + "\nsecond true: " + data.instance(4).classValue());
	}
}