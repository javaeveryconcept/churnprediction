package com.ai.churnprediction.util;

import au.com.bytecode.opencsv.CSVReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class AiUtil {
    public static final String DATASET_PATH = "src/main/resources/WA_Fn-UseC_-Telco-Customer-Churn.csv";

    public static List<Map<String, Object>> loadCsvData() throws IOException {
        List<Map<String, Object>> data = new ArrayList<>();
        try (CSVReader reader = new CSVReader(new FileReader(AiUtil.DATASET_PATH))) {
            String[] headers = reader.readNext();
            String[] row;
            while ((row = reader.readNext()) != null) {
                Map<String, Object> rowMap = new LinkedHashMap<>();
                for (int i = 0; i < headers.length; i++) {
                    rowMap.put(headers[i], row[i]);
                }
                data.add(rowMap);
            }
        }
        return data;
    }
}
