package com.ai.churnprediction.controller;

import com.ai.churnprediction.service.ChurnPrediction;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api/churn")
@Slf4j
public class ChurnPredictionController {
    private final ChurnPrediction churnPredictionService;

    public ChurnPredictionController(ChurnPrediction churnPredictionService) {
        this.churnPredictionService = churnPredictionService;
    }

    @PostMapping("/predict")
    public String predict(@RequestBody Map<String, Object> payload) {
        log.info("Received churn prediction request: {}", payload.get("customerID"));
        try {
            double probability = churnPredictionService.predictChurn(payload);
            String label = probability >= 0.5 ? "Yes" : "No";
            return String.format("Churn Prediction: %s (Probability: %.4f)", label, probability);
        } catch (Exception e) {
            return "Prediction failed: " + e.getMessage();
        }
    }

}
