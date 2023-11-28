import com.knf.dev.demo.dto.ColumnsAndMetricsRequest;
import com.knf.dev.demo.service.MemberPaymentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/memberPayments")
public class MemberPaymentController {

    @Autowired
    private MemberPaymentService memberPaymentService;

    // Other API methods...

    @PostMapping("/columns")
    public ResponseEntity<Map<String, Object>> getMemberPaymentsByColumnsAndMetrics(@RequestBody ColumnsAndMetricsRequest request) {
        try {
            Map<String, Object> response = memberPaymentService.getMemberPaymentsByColumnsAndMetrics(request);
            return new ResponseEntity<>(response, HttpStatus.OK);
        } catch (Exception e) {
            // Handle exceptions appropriately for your application
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }
}
