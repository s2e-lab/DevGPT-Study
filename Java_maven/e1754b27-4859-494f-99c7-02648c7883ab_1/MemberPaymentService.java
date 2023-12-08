import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.stereotype.Service;

@Service
public class MemberPaymentService {

    @Autowired
    private MemberPaymentRepository memberPaymentRepository;

    public Workbook createExcel() {
        List<MemberPayment> memberPayments = memberPaymentRepository.findAll();

        Workbook workbook = new XSSFWorkbook();
        Sheet sheet = workbook.createSheet("MemberPayments");
        Row headerRow = sheet.createRow(0);

        // Headers
        String[] columns = {"Member ID", "Confirmation Number", "Payment Date", "Payment Type", "Payment Amount", "Convenience Fees Amount", "Total Payment Amount"};
        for (int i = 0; i < columns.length; i++) {
            Cell cell = headerRow.createCell(i);
            cell.setCellValue(columns[i]);
        }

        int rowNum = 1;
        for (MemberPayment payment : memberPayments) {
            Row row = sheet.createRow(rowNum++);
            row.createCell(0).setCellValue(payment.getMemberId());
            row.createCell(1).setCellValue(payment.getConfirmationNumber());
            row.createCell(2).setCellValue(payment.getPaymentDate());
            row.createCell(3).setCellValue(payment.getPaymentType());
            row.createCell(4).setCellValue(payment.getPaymentAmount());
            row.createCell(5).setCellValue(payment.getConvenienceFeesAmount());
            row.createCell(6).setCellValue(payment.getTotalPaymentAmount());
        }

        return workbook;
    }
}
