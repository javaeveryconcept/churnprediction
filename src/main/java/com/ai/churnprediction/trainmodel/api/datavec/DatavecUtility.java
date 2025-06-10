package com.ai.churnprediction.trainmodel.api.datavec;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.StringColumnCondition;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.doubletransform.ConvertToDouble;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.api.writable.Text;
import org.datavec.local.transforms.AnalyzeLocal;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class DatavecUtility {

    /**
     * Builds the input schema for the customer churn dataset.
     * This schema defines the structure of the data, including column names and types.
     *
     * @return Schema object representing the input data structure.
     */
    public static Schema buildInputSchema() {
        return new Schema.Builder()
                .addColumnString("customerID") // Will be removed later
                .addColumnCategorical("gender", Arrays.asList("Male", "Female"))
                .addColumnInteger("SeniorCitizen")
                .addColumnCategorical("Partner", Arrays.asList("Yes", "No"))
                .addColumnCategorical("Dependents", Arrays.asList("Yes", "No"))
                .addColumnInteger("tenure")
                .addColumnCategorical("PhoneService", Arrays.asList("Yes", "No"))
                .addColumnCategorical("MultipleLines", Arrays.asList("Yes", "No", "No phone service"))
                .addColumnCategorical("InternetService", Arrays.asList("DSL", "Fiber optic", "No"))
                .addColumnCategorical("OnlineSecurity", Arrays.asList("Yes", "No", "No internet service"))
                .addColumnCategorical("OnlineBackup", Arrays.asList("Yes", "No", "No internet service"))
                .addColumnCategorical("DeviceProtection", Arrays.asList("Yes", "No", "No internet service"))
                .addColumnCategorical("TechSupport", Arrays.asList("Yes", "No", "No internet service"))
                .addColumnCategorical("StreamingTV", Arrays.asList("Yes", "No", "No internet service"))
                .addColumnCategorical("StreamingMovies", Arrays.asList("Yes", "No", "No internet service"))
                .addColumnCategorical("Contract", Arrays.asList("Month-to-month", "One year", "Two year"))
                .addColumnCategorical("PaperlessBilling", Arrays.asList("Yes", "No"))
                .addColumnCategorical("PaymentMethod", Arrays.asList(
                        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"))
                .addColumnDouble("MonthlyCharges")
                .addColumnString("TotalCharges") // may need to handle missing/blank
                .addColumnCategorical("Churn", Arrays.asList("Yes", "No")) // label
                .build();

    }

    /**
     *  Performs statistical analysis on the data (mean, stddev, min, max)  to normalize numeric fields.
     *  Computes the mean, min, max, std for all numeric columns
     * @param cleanedCsv
     * @param preProcess
     * @return
     * @throws IOException
     * @throws InterruptedException
     */
    public static DataAnalysis staticalAnalysis(File cleanedCsv, TransformProcess preProcess) throws IOException, InterruptedException {
        RecordReader preReader = new CSVRecordReader(1, ',');
        preReader.initialize(new FileSplit(cleanedCsv));
        RecordReader preProcessedReader = new TransformProcessRecordReader(preReader, preProcess);
        //Analyze the specified data - returns a DataAnalysis object with summary information about each column
        return AnalyzeLocal.analyze(preProcess.getFinalSchema(), preProcessedReader);
    }


    //Converting TotalCharges to double type
    public static TransformProcess simplifyTotalCharges(Schema schema) {
        return new TransformProcess.Builder(schema)
                .conditionalReplaceValueTransform("TotalCharges", new Text("0"),
                        new StringColumnCondition("TotalCharges", ConditionOp.Equal, "")
                )
                .transform(new ConvertToDouble("TotalCharges"))
                .build();
    }

    /**
     * This builds the final transformation pipeline, including
     *  - Removing customerID
     *  - Converting strings to one-hot encoding or integers
     *  - Normalize numeric values (tenure, MonthlyCharges, TotalCharges)
     * This turns raw CSV data into clean numeric vectors ready for training.
     */
    public static TransformProcess buildFullTransform(Schema schema, DataAnalysis analysis) {
        return new TransformProcess.Builder(schema)
                .removeColumns("customerID")
                .stringToCategorical("gender", Arrays.asList("Male", "Female")).categoricalToOneHot("gender")
                .stringToCategorical("Partner", Arrays.asList("Yes", "No")).categoricalToOneHot("Partner")
                .stringToCategorical("Dependents", Arrays.asList("Yes", "No")).categoricalToOneHot("Dependents")
                .stringToCategorical("PhoneService", Arrays.asList("Yes", "No")).categoricalToOneHot("PhoneService")
                .stringToCategorical("MultipleLines", Arrays.asList("No phone service", "No", "Yes")).categoricalToOneHot("MultipleLines")
                .stringToCategorical("InternetService", Arrays.asList("DSL", "Fiber optic", "No")).categoricalToOneHot("InternetService")
                .stringToCategorical("OnlineSecurity", Arrays.asList("No", "Yes", "No internet service")).categoricalToOneHot("OnlineSecurity")
                .stringToCategorical("OnlineBackup", Arrays.asList("No", "Yes", "No internet service")).categoricalToOneHot("OnlineBackup")
                .stringToCategorical("DeviceProtection", Arrays.asList("No", "Yes", "No internet service")).categoricalToOneHot("DeviceProtection")
                .stringToCategorical("TechSupport", Arrays.asList("No", "Yes", "No internet service")).categoricalToOneHot("TechSupport")
                .stringToCategorical("StreamingTV", Arrays.asList("No", "Yes", "No internet service")).categoricalToOneHot("StreamingTV")
                .stringToCategorical("StreamingMovies", Arrays.asList("No", "Yes", "No internet service")).categoricalToOneHot("StreamingMovies")
                .stringToCategorical("Contract", Arrays.asList("Month-to-month", "One year", "Two year")).categoricalToOneHot("Contract")
                .stringToCategorical("PaperlessBilling", Arrays.asList("Yes", "No")).categoricalToOneHot("PaperlessBilling")
                .stringToCategorical("PaymentMethod", Arrays.asList("Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)")).categoricalToOneHot("PaymentMethod")

                // Standardizes numerical values (mean=0, std=1) to help neural networks learn better
                .normalize("tenure", Normalize.Standardize, analysis)
                .normalize("MonthlyCharges", Normalize.Standardize, analysis)
                .normalize("TotalCharges", Normalize.Standardize, analysis)
                // Final label
                .stringToCategorical("Churn", Arrays.asList("Yes", "No")).categoricalToInteger("Churn")
                //  All columns accounted for: 21 total
                // - customerID (optional to remove separately)
                // - SeniorCitizen, tenure, MonthlyCharges: numerical (left as-is)
                .build();
    }


    /**
     * Generates an HTML report of the data analysis.
     *
     * @param analysis The DataAnalysis object containing statistical information.
     * @return A string containing the HTML report.
     */
    public static void htmlDataAnalysis(DataAnalysis analysis) throws Exception {
        HtmlAnalysis.createHtmlAnalysisFile(analysis,new File("analysis.html"));
    }


    public static DataSetIterator applyFinalTransform(TransformProcess fullTransform, File cleanedCsv) throws IOException, InterruptedException {
        RecordReader finalReader = new CSVRecordReader(1, ',');
        finalReader.initialize(new FileSplit(cleanedCsv));
        RecordReader transformedReader = new TransformProcessRecordReader(finalReader, fullTransform);

        int batchSize = 1000;
        int labelIndex = fullTransform.getFinalSchema().getColumnNames().indexOf("Churn");
        int numClasses = 1; // Binary classification
        return new RecordReaderDataSetIterator(transformedReader, batchSize, labelIndex, numClasses);
    }
}
