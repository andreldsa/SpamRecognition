package classificador;

import java.io.File;

import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;

public class Classificador {

	private static final String VERBOSE = "-v";

	public static void main(String[] args) {
		String tecnic = args[0];
		String trainPath = args[1];
		String testPath = args[2];
		if (args[0].equals(VERBOSE)) {
			tecnic = args[1];
			trainPath = args[2];
			testPath = args[3];
		}
		try {
			System.out.println("Processando dados...");
			DataLearner learner = new DataLearner(trainPath, tecnic);

			FilteredClassifier classifier = learner.learn();

			if (args[0].equals(VERBOSE)) {
				classificaEmail(classifier, testPath);
			}
			System.out.println("\n"+learner.getSpamDetails());
		} catch (Exception e) {
			e.printStackTrace();
			System.err.println(e.getMessage());
			System.exit(1);
		}
	}

	public static void classificaEmail(FilteredClassifier classifier,
			String diretorio) throws Exception {
		Instances test = DataLearner.createARFF(new File(diretorio));
		test.setClassIndex(2);
		classifier.buildClassifier(test);
		for (int i = 0; i < test.numInstances(); i++) {
			double pred = classifier.classifyInstance(test.instance(i));
			String name = test.instance(i).stringValue(1);
			String predict = test.classAttribute().value((int) pred);
			System.out.println(String.format("%s: %s", name, predict));
		}
	}
}
