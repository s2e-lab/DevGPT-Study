package com.knf.dev.demo.dto;

import java.util.List;

public class ColumnsAndMetricsRequest {
    private List<String> headers;
    private List<String> metrics;

    // Getters and setters

    public List<String> getHeaders() {
        return headers;
    }

    public void setHeaders(List<String> headers) {
        this.headers = headers;
    }

    public List<String> getMetrics() {
        return metrics;
    }

    public void setMetrics(List<String> metrics) {
        this.metrics = metrics;
    }
}