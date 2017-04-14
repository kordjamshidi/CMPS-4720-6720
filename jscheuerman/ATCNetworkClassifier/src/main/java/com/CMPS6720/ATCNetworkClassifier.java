package com.CMPS6720;

import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class ATCNetworkClassifier
{
    private static Logger log = LoggerFactory.getLogger(ATCNetworkClassifier.class);

    private static int numInputs = 164;
    private static int outputNum = 2;
    private static long seed = 6;
    private static ArrayList<String[]> results = new ArrayList<String[]>();

    private static double bestF1 = 0.0;
    private static int bestIteration = 1;
    private static double bestRegularization = 0.000001;
    private static MultiLayerNetwork bestModel;

    public static void main( String[] args ) throws Exception
    {
        //Prepare data
        RecordReader recordReader = new CSVRecordReader(1, ",");
        File file = new File("trainingandvalidation.csv");

        RecordReader headerReader = new CSVRecordReader(0, ",");
        headerReader.initialize(new FileSplit(file));
        List columns = headerReader.next();

        //Read first line of file in for tracking weights

        File testFile = new File("test.csv");

        recordReader.initialize(new FileSplit(file));
        int labelIndex = 0;
        int numClasses = 2;
        int batchSize = 80;
        int counter = 0;
        int maxCount = 10;

        int maxIterations = 25;
        Double[] regularizationTerms = {0.000001, 0.000002, 0.000004, 0.000008, 0.00001, 0.00002, 0.00004, 0.00008, 0.00016, 0.00032, 0.00064, 0.00128, 0.00256, 0.00512, 0.01024, 0.02048, 0.04090, 0.08180, 0.16360};


        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet allData = iterator.next();

        recordReader.initialize(new FileSplit(testFile));
        DataSetIterator iteratorTest = new RecordReaderDataSetIterator(recordReader,batchSize,labelIndex,numClasses);
        DataSet testData = iteratorTest.next();

        while (counter < maxCount) {
            allData.shuffle();
            SplitTestAndTrain validateAndTrain = allData.splitTestAndTrain(0.75);
            DataSet trainingData = validateAndTrain.getTrain();
            DataSet validateData = validateAndTrain.getTest();

            DataNormalization normalizer = new NormalizerStandardize();
            normalizer.fit(allData);
            normalizer.transform(trainingData);
            normalizer.transform(validateData);
            normalizer.transform(testData);
            //regularization terms to try
            trainAndValidateModels(trainingData, validateData, maxIterations, regularizationTerms);
            outputBestModelResults(testData);

            //output to file
            BufferedWriter br = new BufferedWriter(new FileWriter("bestweights_" + counter + ".csv"));
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < columns.size(); i++) {
                sb.append(columns.get(i).toString());
                sb.append(",");
                sb.append(bestModel.params().getColumn(i) + "\n");
            }

            br.write(sb.toString());
            br.close();

            //output to file
            br = new BufferedWriter(new FileWriter("results_" + counter + ".csv"));
            sb = new StringBuilder();
            for (String[] outputRow : results) {
                for (String s : outputRow) {
                    sb.append(s);
                    sb.append(",");
                }
                sb.append("\n");
            }

            br.write(sb.toString());
            br.close();
            counter++;
        }
    }

    static void trainAndValidateModels(DataSet trainingData, DataSet validateData, int maxIterations, Double[] regularizationTerms) {

        int i = 0;
        Integer j = 0;
        ArrayList<String []> outputResults = new ArrayList<String[]>();

        String [] row = {"Model type","Regularization","Iteration","Accuracy","L0 Precision","L1 Precision","L0 Recall","L1 Recall","L0 F1","L1 F1"};
        results.add(row);

        while (i < regularizationTerms.length) {
            j = 0;
            while (j < maxIterations) {
                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .iterations(j)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.ZERO)
                        .learningRate(0.1)
                        .regularization(true).l1(regularizationTerms[i])
                        .list()
                        .layer(0, new OutputLayer.Builder(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                                .activation(Activation.SOFTSIGN)
                                .nIn(numInputs).nOut(outputNum).build())
                        .build();


                //run the model
                MultiLayerNetwork model = new MultiLayerNetwork(conf);
                model.init();
                model.setListeners(new ScoreIterationListener(1));

                model.fit(trainingData);


                //evaluate the model on the test set
                Evaluation eval = new Evaluation(3);

                INDArray output = model.output(validateData.getFeatureMatrix());
                eval.eval(validateData.getLabels(), output);
                log.info(eval.stats());
                double evalF1 = eval.f1(1);

                if (evalF1 > bestF1) {
                    bestF1 = evalF1;
                    bestIteration = j;
                    bestRegularization = regularizationTerms[i];
                    bestModel = model;
                }

                String[] newRow = {
                        "All Data",
                        regularizationTerms[i].toString(),
                        j.toString(),
                        Double.toString(eval.accuracy()),
                        Double.toString(eval.precision(0)),
                        Double.toString(eval.precision(1)),
                        Double.toString(eval.recall(0)),
                        Double.toString(eval.recall(1)),
                        Double.toString(eval.f1(0)),
                        Double.toString(eval.f1(1))};
                results.add(newRow);
                j++;
            }
            i++;
        }
    }

    static void outputBestModelResults(DataSet testData) {
        bestModel.fit(testData);
        System.out.println("Best iteration #:" + bestIteration);
        System.out.println("Best regularization term: " + bestRegularization);
        System.out.println("Best F1: " + bestF1);

        //evaluate the model on the test set
        Evaluation eval = new Evaluation(3);
        INDArray output = bestModel.output(testData.getFeatureMatrix());
        eval.eval(testData.getLabels(), output);
        log.info(eval.stats());
    }
}
