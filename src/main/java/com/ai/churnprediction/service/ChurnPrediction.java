package com.ai.churnprediction.service;

import com.ai.churnprediction.util.AiUtil;
import jakarta.annotation.PostConstruct;
import jakarta.servlet.http.HttpServletResponse;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Service
@Slf4j
public class ChurnPrediction {
    private MultiLayerNetwork model;
    private TransformProcess transformProcess;
    private Schema inputSchema;
    @Getter
    private List<Map<String, Object>> fullData = new ArrayList<>();

    @PostConstruct
    public void init() throws Exception {
        // Load full dataset for UI
        fullData = AiUtil.loadCsvData();
        //
        // Load trained model
        model = ModelSerializer.restoreMultiLayerNetwork(new File("churn-model.zip"));
        log.info("Model loaded successfully! {}", model.summary());
        // Load TransformProcess from JSON
        String json = Files.readString(new File("transformProcess.json").toPath());
        transformProcess = TransformProcess.fromJson(json);

        // Save schema for mapping incoming JSON
        inputSchema = transformProcess.getInitialSchema();
        log.info("TransformProcess loaded successfully! {}", transformProcess.getFinalSchema());
    }

    public double predictChurn(Map<String, Object> payload) {
        if (payload.get("TotalCharges") == null || payload.get("TotalCharges").toString().trim().isEmpty()) {
            payload.put("TotalCharges", 0.0);
        }

        // Step 1: Convert JSON payload into List<Writable> using schema
        List<Writable> record = convertPayloadToWritable(payload);

        // Step 2: Apply the same TransformProcess as training
        double[] features = convertRecordToFeatures(record);

        // Step 3: Predict
        return predictChurn(features);
    }

    public void getAllPredictions(HttpServletResponse response) throws IOException {
        // Write CSV content
        PrintWriter writer = response.getWriter();
        writer.println("CustomerID,Prediction,Probability");

        fullData.forEach(stringObjectMap -> {
            double prob = predictChurn(stringObjectMap);
            String label = prob >= 0.5 ? "Yes" : "No";
            writer.printf("%s,%s,%.4f%n", stringObjectMap.get("customerID").toString(), label, prob);
        });
        writer.flush();
        writer.close();
    }

    private List<Writable> convertPayloadToWritable(Map<String, Object> payload) {
        List<Writable> record = new ArrayList<>();
        for (String col : inputSchema.getColumnNames()) {
            Object value = payload.get(col);
            if (value == null || value.toString().trim().isEmpty()) {
                throw new RuntimeException("Missing or empty value for input column: " + col);
            }
            // Even if it's not used, still include it in the raw record
            record.add(new Text(value.toString()));
        }
        return record;
    }

    private double[] convertRecordToFeatures(List<Writable> record) {
        List<Writable> transformed = transformProcess.execute(record);
        // Remove `churn` if it's a label or not needed for prediction
        transformed.removeLast();

        //Convert List<Writable> to INDArray
        return transformed.stream()
                .mapToDouble(w -> Double.parseDouble(w.toString()))
                .toArray();
    }

    private double predictChurn(double[] features) {
        // Create INDArray with shape [1, numFeatures]
        INDArray input = Nd4j.create(features).reshape(1, features.length);
        return model.output(input).getDouble(0);
    }
}
