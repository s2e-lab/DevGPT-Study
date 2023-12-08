import com.itextpdf.text.*;
import com.itextpdf.text.pdf.*;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.pdmodel.common.PDRectangle;
import org.apache.pdfbox.pdmodel.PDPageContentStream;
import org.apache.pdfbox.pdmodel.font.PDType1Font;
import org.springframework.stereotype.Service;

import java.io.ByteArrayOutputStream;
import java.util.List;

@Service
public class MemberPaymentService {

    @Autowired
    private MemberPaymentRepository memberPaymentRepository;

    public byte[] createPdfWithItext() throws Exception {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        Document document = new Document();
        PdfWriter.getInstance(document, out);
        document.open();

        PdfPTable table = new PdfPTable(7); // 7 columns
        String[] columns = {"Member ID", "Confirmation Number", "Payment Date", "Payment Type", "Payment Amount", "Convenience Fees Amount", "Total Payment Amount"};
        
        for (String column : columns) {
            table.addCell(column);
        }
        
        List<MemberPayment> payments = memberPaymentRepository.findAll();
        for (MemberPayment payment : payments) {
            table.addCell(payment.getMemberId());
            table.addCell(payment.getConfirmationNumber());
            table.addCell(payment.getPaymentDate());
            table.addCell(payment.getPaymentType());
            table.addCell(String.valueOf(payment.getPaymentAmount()));
            table.addCell(String.valueOf(payment.getConvenienceFeesAmount()));
            table.addCell(String.valueOf(payment.getTotalPaymentAmount()));
        }

        document.add(table);
        document.close();

        return out.toByteArray();
    }

    public byte[] createPdfWithPdfbox() throws Exception {
        PDDocument doc = new PDDocument();
        PDPage page = new PDPage(PDRectangle.A4);
        doc.addPage(page);
        PDPageContentStream contentStream = new PDPageContentStream(doc, page);

        contentStream.setFont(PDType1Font.HELVETICA_BOLD, 12);
        String[] columns = {"Member ID", "Confirmation Number", "Payment Date", "Payment Type", "Payment Amount", "Convenience Fees Amount", "Total Payment Amount"};

        int yPosition = 750; // start from top
        int tableRowHeight = 20;
        
        for (String column : columns) {
            contentStream.beginText();
            contentStream.newLineAtOffset(50, yPosition);
            contentStream.showText(column);
            contentStream.endText();
            yPosition -= tableRowHeight;
        }
        
        List<MemberPayment> payments = memberPaymentRepository.findAll();
        for (MemberPayment payment : payments) {
            contentStream.beginText();
            contentStream.newLineAtOffset(50, yPosition);
            contentStream.showText(payment.getMemberId() + " " + payment.getConfirmationNumber() + " " + payment.getPaymentDate() + " " + payment.getPaymentType() + " " + String.valueOf(payment.getPaymentAmount()) + " " + String.valueOf(payment.getConvenienceFeesAmount()) + " " + String.valueOf(payment.getTotalPaymentAmount()));
            contentStream.endText();
            yPosition -= tableRowHeight;
        }

        contentStream.close();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        doc.save(out);
        doc.close();

        return out.toByteArray();
    }
}
