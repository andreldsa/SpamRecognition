package classificador;

import java.io.File;
import java.io.IOException;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.core.converters.TextDirectoryLoader;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class DataLearner {
	
	private Instances dataSet;
	
	private Evaluation evaluation;

	private String tecnic;
	
	public DataLearner(String filePath, String tecnic) throws IOException {
		this.dataSet = createARFF(new File(filePath));
		this.tecnic = tecnic;
	}

	public static Instances createARFF(File file) throws IOException {
		TextDirectoryLoader loader = new TextDirectoryLoader();
		loader.setOutputFilename(true);
		loader.setDirectory(file);
		Instances dataSet = loader.getDataSet();
		return dataSet;
	}
	
	public void evaluate(FilteredClassifier classifier) {
		try {
			evaluation = new Evaluation(dataSet);
			evaluation.crossValidateModel(classifier, dataSet, 4, new Random(1));
		}
		catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
		}
	}
	
	public FilteredClassifier learn() {
		try {
			dataSet.setClassIndex(2);
			StringToWordVector filter = new StringToWordVector();
			filter.setAttributeIndices("first-last");
			
			FilteredClassifier classifier = new FilteredClassifier();
			classifier.setFilter(filter);
			if (tecnic.equals("SMO")) {
				classifier.setClassifier(new SMO());
			} else if (tecnic.equals("NaiveBayes")) {
				classifier.setClassifier(new NaiveBayes());
			} else {
				System.err.println("Unknown Tecnic. Use SMO or NaiveBayes!");
				System.exit(1);
			}
			classifier.buildClassifier(dataSet);

			evaluate(classifier);
			
			return classifier;
		}
		catch (Exception e) {
			e.printStackTrace();
			System.exit(1);
			return null;
		}
	}
	
	public String getSummary() {
		return evaluation.toSummaryString();
	}
	
	public String getAccuracy() throws Exception {
		return evaluation.toClassDetailsString();
	}
	
	public String getSpamDetails() {
		Double recall = evaluation.recall(0);
		Double precision = evaluation.precision(0);
		Double fMeasure = evaluation.fMeasure(0);
		return String.format("precision: %.3f\nrecall: %.3f\nf-measure: %.3f", precision, recall, fMeasure);
	}
}
