@RestController
@RequestMapping("/api/v1/memberPayments")
public class MemberPaymentController {

    @Autowired
    private MemberPaymentService memberPaymentService;

    @GetMapping("/download/pdf/itext")
    public ResponseEntity<ByteArrayResource> downloadPdfWithItext() {
        try {
            byte[] pdfContent = memberPaymentService.createPdfWithItext();
            ByteArrayResource resource = new ByteArrayResource(pdfContent);

            HttpHeaders headers = new HttpHeaders();
            headers.add(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=memberPayments-itext.pdf");

            return ResponseEntity.ok()
                    .headers(headers)
                    .contentLength(resource.contentLength())
                    .contentType(MediaType.parseMediaType("application/pdf"))
                    .body(resource);

        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }

    @GetMapping("/download/pdf/pdfbox")
    public ResponseEntity<ByteArrayResource> downloadPdfWithPdfbox() {
        try {
            byte[] pdfContent = memberPaymentService.createPdfWithPdfbox();
            ByteArrayResource resource = new ByteArrayResource(pdfContent);

            HttpHeaders headers = new HttpHeaders();
            headers.add(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=memberPayments-pdfbox.pdf");

            return ResponseEntity.ok()
                    .headers(headers)
                    .contentLength(resource.contentLength())
                    .contentType(MediaType.parseMediaType("application/pdf"))
                    .body(resource);

        } catch (Exception e) {
            return new ResponseEntity<>(HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }
}
