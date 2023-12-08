public class FileFrameController {
    public String getFileURL() {
        Id recordId = ApexPages.currentPage().getParameters().get('recordId');
        ContentDocumentLink link = [SELECT ContentDocumentId FROM ContentDocumentLink WHERE LinkedEntityId = :recordId LIMIT 1];
        String fileURL = '/sfc/servlet.shepherd/document/download/' + link.ContentDocumentId;
        return fileURL;
    }
}
