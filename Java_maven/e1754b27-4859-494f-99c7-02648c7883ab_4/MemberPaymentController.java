package com.knf.dev.demo.controller;

import com.knf.dev.demo.dto.ColumnsRequest;
import com.knf.dev.demo.service.MemberPaymentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/memberPayments")
public class MemberPaymentController {

    @Autowired
    MemberPaymentService memberPaymentService;

    @PostMapping("/columns")
    public ResponseEntity<List<Map<String, Object>>> getMemberPaymentsByColumns(@RequestBody ColumnsRequest request) {
        try {
            List<Map<String, Object>> payments = memberPaymentService.getMemberPaymentsByColumns(request.getHeaders());
            return new ResponseEntity<>(payments, HttpStatus.OK);
        } catch (Exception e) {
            throw new UnKnownException(e.getMessage());
        }
    }
}
