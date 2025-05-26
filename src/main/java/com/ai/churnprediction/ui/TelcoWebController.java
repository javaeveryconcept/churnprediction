package com.ai.churnprediction.ui;

import com.ai.churnprediction.service.ChurnPrediction;
import jakarta.servlet.http.HttpServletResponse;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Controller
@Slf4j
public class TelcoWebController {

    @Autowired
    private ChurnPrediction predictionService;

    @PostMapping("/predict")
    @ResponseBody
    public String predict(@RequestParam String customerID) {
        log.info("Predicting churn for customerID: {}", customerID);
        Optional<Map<String, Object>> customer = predictionService.getFullData().stream()
                .filter(row -> row.get("customerID").equals(customerID))
                .findFirst();

        if (customer.isEmpty()) return "Customer not found.";

        double prob = predictionService.predictChurn(customer.get());
        String label = prob >= 0.5 ? "Yes" : "No";

        return String.format("Churn: %s (%.4f)", label, prob);
    }

    @GetMapping("/index")
    public String index(Model model,
                        @RequestParam(name = "search", required = false) String search,
                        @RequestParam(defaultValue = "0") int page,
                        @RequestParam(defaultValue = "10") int size) {
        log.info("Loading index page with search: {}", search);

        List<Map<String, Object>> filtered = predictionService.getFullData();
        if (search != null && !search.isEmpty()) {
            filtered = filtered.stream()
                    .filter(row -> row.get("customerID").toString().contains(search))
                    .toList();
        }

        int totalRecords = filtered.size();
        int fromIndex = Math.min(page * size, totalRecords);
        int toIndex = Math.min(fromIndex + size, totalRecords);
        List<Map<String, Object>> pageData = filtered.subList(fromIndex, toIndex);

        int totalPages = (int) Math.ceil((double) totalRecords / size);

        model.addAttribute("data", pageData);
        model.addAttribute("search", search);
        model.addAttribute("currentPage", page);
        model.addAttribute("pageSize", size);
        model.addAttribute("totalPages", totalPages);
        model.addAttribute("pageSizes", List.of(10, 20, 50, 100));
        model.addAttribute("totalRecords", totalRecords);
        model.addAttribute("paginationRange", getPaginationRange(page, totalPages));
        return "index";
    }

    @GetMapping("/download-report")
    public void downloadCsvReport(HttpServletResponse response) throws IOException {
        // Set response headers
        response.setContentType("text/csv");
        response.setHeader("Content-Disposition", "attachment; filename=predict_report.csv");
        // Loop through data with predictions (replace with your actual logic)
        predictionService.getAllPredictions(response); // create this service method
    }

    private List<Integer> getPaginationRange(int currentPage, int totalPages) {
        int start = Math.max(0, currentPage - 2);
        int end = Math.min(totalPages - 1, currentPage + 2);
        List<Integer> range = new ArrayList<>();

        if (start > 0) range.add(0);         // always show first
        if (start > 1) range.add(-1);        // -1 means ellipsis

        for (int i = start; i <= end; i++) {
            range.add(i);
        }

        if (end < totalPages - 2) range.add(-2); // -2 means ellipsis
        if (end < totalPages - 1) range.add(totalPages - 1); // always show last

        return range;
    }

}

