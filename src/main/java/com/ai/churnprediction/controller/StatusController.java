package com.ai.churnprediction.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class StatusController {
    @GetMapping("/")
    public String status() {
        return "Churn Prediction Service is running!";
    }
}
