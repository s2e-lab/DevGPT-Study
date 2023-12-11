package com.knf.dev.demo.dto;

import com.knf.dev.demo.entity.MemberPayment;
import com.knf.dev.demo.exception.MemberNotFound;
import com.knf.dev.demo.repository.MemberPaymentRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
@Service
public class MemberPaymentService {

    @Autowired
    MemberPaymentRepository memberPaymentRepository;

    public MemberPayment createMemberPayment(MemberPayment memberPayment) {
        return memberPaymentRepository.save(memberPayment);
    }

    public List<MemberPayment> getAllMemberPayments() {
        return memberPaymentRepository.findAll();
    }

    public MemberPayment getMemberPaymentById(String id) {
        Optional<MemberPayment> memberPayment = memberPaymentRepository.findById(id);
        return memberPayment.orElseThrow(() -> new MemberNotFound("Member Payment not found for this id :: " + id));
    }

    public MemberPayment updateMemberPayment(String id, MemberPayment memberPaymentDetails) {
        MemberPayment memberPayment = getMemberPaymentById(id);

        memberPayment.setConfirmationNumber(memberPaymentDetails.getConfirmationNumber());
        memberPayment.setPaymentDate(memberPaymentDetails.getPaymentDate());
        memberPayment.setPaymentType(memberPaymentDetails.getPaymentType());
        memberPayment.setPaymentAmount(memberPaymentDetails.getPaymentAmount());
        memberPayment.setConvenienceFeesAmount(memberPaymentDetails.getConvenienceFeesAmount());
        memberPayment.setTotalPaymentAmount(memberPaymentDetails.getTotalPaymentAmount());

        return memberPaymentRepository.save(memberPayment);
    }

    public void deleteMemberPayment(String id) {
        MemberPayment memberPayment = getMemberPaymentById(id);
        memberPaymentRepository.delete(memberPayment);
    }

    public Map<String, Object> getMemberPaymentsByColumnsAndMetrics(ColumnsAndMetricsRequest request) {
        List<MemberPayment> payments = memberPaymentRepository.findAll();
        Map<String, Object> response = new HashMap<>();

        // Calculate metrics
        BigDecimal totalPaymentAmount = BigDecimal.ZERO;
        BigDecimal totalConvenienceAmount = BigDecimal.ZERO;

        for (MemberPayment payment : payments) {
            totalPaymentAmount = totalPaymentAmount.add(payment.getPaymentAmount());
            totalConvenienceAmount = totalConvenienceAmount.add(payment.getConvenienceFeesAmount());
        }

        // Add metrics to response if requested
        if (request.getMetrics() != null) {
            if (request.getMetrics().contains("totalpaymentsamount")) {
                response.put("totalpaymentsamount", totalPaymentAmount);
            }
            if (request.getMetrics().contains("totalconvenienceamount")) {
                response.put("totalconvenienceamount", totalConvenienceAmount);
            }
        }

        // Include requested columns in response
        List<Map<String, Object>> paymentList = new ArrayList<>();
        for (MemberPayment payment : payments) {
            Map<String, Object> paymentMap = new HashMap<>();
            if (request.getHeaders().contains("memberId")) {
                paymentMap.put("memberId", payment.getMemberId());
            }
            if (request.getHeaders().contains("confirmationNumber")) {
                paymentMap.put("confirmationNumber", payment.getConfirmationNumber());
            }
            if (request.getHeaders().contains("paymentDate")) {
                paymentMap.put("paymentDate", payment.getPaymentDate());
            }
            if (request.getHeaders().contains("paymentType")) {
                paymentMap.put("paymentType", payment.getPaymentType());
            }
            if (request.getHeaders().contains("paymentAmount")) {
                paymentMap.put("paymentAmount", payment.getPaymentAmount());
            }
            if (request.getHeaders().contains("convenienceFeesAmount")) {
                paymentMap.put("convenienceFeesAmount", payment.getConvenienceFeesAmount());
            }
            if (request.getHeaders().contains("totalPaymentAmount")) {
                paymentMap.put("totalPaymentAmount", payment.getTotalPaymentAmount());
            }
            paymentList.add(paymentMap);
        }

        response.put("payments", paymentList);

        return response;
    }

}