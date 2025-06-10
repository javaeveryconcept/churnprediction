package com.ai.churnprediction.trainmodel;

import com.ai.churnprediction.trainmodel.api.datavec.DatavecUtility;
import com.ai.churnprediction.trainmodel.api.deeplearning4j.DeepLearning4JUtility;
import com.ai.churnprediction.trainmodel.api.nd4j.ND4JUtility;

import com.ai.churnprediction.util.AiUtil;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.InferredSchema;
import org.datavec.api.transform.schema.Schema;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.util.Arrays;

/**
 * This class handles the complete machine learning pipeline using the Kaggle Telco Customer Churn dataset with Java + DL4J.
 * 1. Cleans the raw CSV data
 * 2. Transforms and prepares the data for AI training
 * 3. Builds and trains a neural network
 * 4. Saves the model and transformation process for future predictions
 */
public class TrainModel {

    public static void main(String[] args) throws Exception {
        // Step 0: Clean the CSV file
        File cleanedCsv = cleanCsvFile();

        // Step 1: Define schema
        Schema inputSchema = new InferredSchema(AiUtil.DATASET_PATH).build();
        System.out.println("Input Schema: " + inputSchema);

        // Step 2: Fix TotalCharges column before analysis
        TransformProcess preProcess = DatavecUtility.simplifyTotalCharges(inputSchema);

        // Step 3: Analyze data for normalization
        DataAnalysis analysis = DatavecUtility.staticalAnalysis(cleanedCsv,preProcess);
        //Visualization
        DatavecUtility.htmlDataAnalysis(analysis);
        
        // Step 4: Full TransformProcess including encoding and normalization
        TransformProcess fullTransform = DatavecUtility.buildFullTransform(preProcess.getFinalSchema(), analysis);
        
        // Step 5: Apply final transform to get DataSet
        //Split data: 80% train, 20% test
        SplitTestAndTrain split = ND4JUtility.getDataSet(fullTransform,cleanedCsv).splitTestAndTrain(0.8);
        DataSet trainData = split.getTrain();
        DataSet testData = split.getTest();

        // Step 6: Configure model and Build Neural Network
        MultiLayerNetwork model = DeepLearning4JUtility.configureModel(trainData);

        // Step 7: Train the model
        trainModel(trainData,model);

        // Step 8: Evaluate on test data
        testAndEvaluateModel(testData,model);

        // Step 9: Save model and transform. We can reuse the model later to make predictions without retraining.
        File modelFile  = saveModelAndTransformProcess(model,fullTransform);

        // 10. Make a prediction with loaded model (from disk) (using test features or to demonstrate loading)
        modelDemonstration(modelFile,testData);

        // Clean up
        if(cleanedCsv.exists()) {
            cleanedCsv.delete();
        }
    }

    
    private static void modelDemonstration(File modelFile, DataSet testData) throws IOException {
        MultiLayerNetwork loadedModel = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        INDArray prediction = loadedModel.output(testData.getFeatures());
        System.out.println("Prediction example (first 5 rows):");
        for (int i = 0; i < 5; i++) {
            System.out.println(prediction.getRow(i));
        }
    }

    private static void testAndEvaluateModel(DataSet testData, MultiLayerNetwork model) {
        INDArray output = model.output(testData.getFeatures());
        Evaluation eval = new Evaluation();
        eval.eval(testData.getLabels(), output);
        System.out.println("Evaluation stats:");
        System.out.println(eval.stats());
    }

    //One pass through the whole dataset. One complete pass through the training data. More epochs = more learning (but risk of overfitting).
    private static void trainModel(DataSet trainData, MultiLayerNetwork model) {
        int nEpochs = 10;
        for (int i = 0; i < nEpochs; i++) {
            System.out.println("Epoch " + (i + 1) + " started");
            model.fit(trainData);  // Adjusts weights to reduce prediction error
            System.out.println("Epoch " + (i + 1) + " completed");
        }
    }




    /**
     * Saves the trained model (churn-model.zip)
     * Saves the full data transformation pipeline (transformProcess.json)
     * This lets you reuse the model and transformation later for prediction.
     */
    private static File saveModelAndTransformProcess(MultiLayerNetwork model,TransformProcess fullTransform) throws IOException {
        File modelFile = new File("churn-model.zip");
        File tranfromProcessFile= new File("transformProcess.json");
        if(modelFile.exists()) {
            modelFile.delete();
        }
        if(tranfromProcessFile.exists()) {
            tranfromProcessFile.delete();
        }
        ModelSerializer.writeModel(model, modelFile, true);
        Files.writeString(tranfromProcessFile.toPath(), fullTransform.toJson());
        System.out.println("Model training complete and saved!");
        return modelFile;
    }




    /**
     *   Cleans the CSV file by replacing empty TotalCharges with 0.0
     */
    private static File cleanCsvFile() throws IOException {
        File cleanedFile = new File(System.getProperty("user.dir"), "cleaned_telco.csv");
        if (cleanedFile.exists()) {
            cleanedFile.delete();
        }
        try (BufferedReader reader = new BufferedReader(new FileReader(AiUtil.DATASET_PATH));
             BufferedWriter writer = new BufferedWriter(new FileWriter(cleanedFile))) {

            String header = reader.readLine();
            writer.write(header);
            writer.newLine();

            int totalChargesIndex = Arrays.asList(header.split(",")).indexOf("TotalCharges");

            String line;
            while ((line = reader.readLine()) != null) {
                String[] tokens = line.split(",", -1);
                if (tokens.length > totalChargesIndex && tokens[totalChargesIndex].trim().isEmpty()) {
                    tokens[totalChargesIndex] = "0.0";
                }
                writer.write(String.join(",", tokens));
                writer.newLine();
            }
        }
        return cleanedFile;
    }
}
